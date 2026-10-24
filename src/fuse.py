"""Fuse N GRU checkpoints into ONE graph whose output is exactly the mean of the members.

The members may differ in hidden width, --diff, --proj, --skip, --dz and --state-reset (same number of layers, no
--ema or --time inputs). Each GRU layer becomes one wide nn.GRU: layer 0 reads the concatenation of the members' own
inputs (raw row, or their projection of [row, row - prev]) with block-structured input weights; deeper layers and
all recurrent weights are block-diagonal. Each member keeps its own input transform and head (dz: normalisation,
GELU+LayerNorm projection, MLP head); the prediction is the mean of the member heads. A member with --state-reset zeroes only its own slice of the hidden state ("reset_slices" in meta.json).

One wide op is faster than N ops in one graph: 256-wide 21.7 us/call vs 25.9 for two 128 ops on the M4
(docs/archive/fable-ideas.md round 2). fused s1+s2 scored 0.68588 held-out, 0.65625 public, 46 min on the platform.

    python src/fuse.py out/fused_x out/final_s1/best.pt out/day_dozken/best.pt [...]
"""
import json, os, shutil, sys
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from model import NF, Net, model_cfg, make_feats  # noqa: E402

out, paths = sys.argv[1], sys.argv[2:]
cks = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
mcs = [model_cfg(c["cfg"]) for c in cks]
L = mcs[0]["layers"]
for mc in mcs:
    assert mc["layers"] == L and not mc["ema"] and not mc["time"], f"unsupported member {mc}"
nets = []
for c, mc in zip(cks, mcs):
    n = Net(mc); n.load_state_dict(c["model"]); n.eval(); nets.append(n)
M = len(nets)
Hs = [mc["hidden"] for mc in mcs]; HT = sum(Hs); offs = np.cumsum([0] + Hs)
any_diff = any(mc["diff"] for mc in mcs)


class Fused(nn.Module):
    def __init__(self):
        super().__init__()
        self.nets = nn.ModuleList(nets)            # members' own input transforms and heads (GRUs replaced below)
        in_dims = [mc["proj"] or NF * (1 + int(mc["diff"])) for mc in mcs]
        self.in_offs = np.cumsum([0] + in_dims)
        self.grus = nn.ModuleList([nn.GRU(int(self.in_offs[-1]) if i == 0 else HT, HT, batch_first=True) for i in range(L)])
        with torch.no_grad():
            for i, g in enumerate(self.grus):
                for p in g.parameters():
                    p.zero_()
                for m, n in enumerate(nets):
                    src, H, o = n.grus[i], Hs[m], int(offs[m])
                    for k in range(3):                      # gates r, z, n
                        rows = slice(k * HT + o, k * HT + o + H); srows = slice(k * H, (k + 1) * H)
                        cols = slice(int(self.in_offs[m]), int(self.in_offs[m + 1])) if i == 0 else slice(o, o + H)
                        g.weight_ih_l0[rows, cols] = src.weight_ih_l0[srows]
                        g.weight_hh_l0[rows, o:o + H] = src.weight_hh_l0[srows]
                        g.bias_ih_l0[rows] = src.bias_ih_l0[srows]
                        g.bias_hh_l0[rows] = src.bias_hh_l0[srows]

    def member_in(self, n, mc, x, d):
        """The input member n feeds its first GRU (and what its head reads besides the GRU output)."""
        if mc.get("dz"):
            xs = ((x - n.mean) / n.std).clamp(-10.0, 10.0)
            return n.inproj(torch.cat([xs, d / n.std], -1) if mc["diff"] else xs)
        f = torch.cat([x, d], -1) if mc["diff"] else x
        return n.proj(f) if n.proj is not None else f

    def member_head(self, n, mc, o, u, x):
        if mc.get("dz"):
            return n.head(n.head_norm(torch.cat([o, u], -1)))[..., :2]     # [:2]: an --aux member's extra outputs are training-only
        return n.head(torch.cat([o, x], -1) if mc["skip"] else o)[..., :2]

    def forward(self, x, d, hs):
        """x: raw rows (B, T, NF); d: x - previous row; hs: per-layer fused hidden states."""
        us = [self.member_in(n, mc, x, d) for n, mc in zip(self.nets, mcs)]
        o, new = torch.cat(us, -1), []
        for g, h in zip(self.grus, hs):
            o, h = g(o, h); new.append(h)
        preds = [self.member_head(n, mc, o[..., int(off):int(off) + H], u, x)
                 for n, mc, u, off, H in zip(self.nets, mcs, us, offs, Hs)]
        return sum(preds) / M, new


class Step(nn.Module):
    def __init__(self, f):
        super().__init__(); self.f = f

    def forward(self, x, *states):
        hs = list(states[:L])
        if any_diff:
            prev = states[L]
            p, new = self.f(x, x - prev, hs)
            return (p, *new, x)
        p, new = self.f(x, torch.zeros_like(x), hs)
        return (p, *new)


fused = Fused().eval()

# parity on real held-out rows: fused output == mean of the members, through the members' own Net.forward
X = np.load(os.path.join(os.path.dirname(__file__), "..", "out", "data", "valid_X.npy"), mmap_mode="r")
x = torch.from_numpy(np.asarray(X[:2, :2500], dtype=np.float32))
with torch.no_grad():
    ref = 0
    for n, mc in zip(nets, mcs):          # each member with its own resets, as solution.py runs it
        f = torch.from_numpy(make_feats(x.numpy(), mc)); hs = [torch.zeros(1, 2, mc["hidden"]) for _ in range(L)]
        R = mc["state_reset"] or 10 ** 9; outs = []
        for c in range(0, x.shape[1], R):
            p, hs = n(f[:, c:c + R], [torch.zeros_like(h) for h in hs] if c else hs); outs.append(p[..., :2])
        ref = ref + torch.cat(outs, 1) / M
    d = torch.zeros_like(x); d[:, 1:] = x[:, 1:] - x[:, :-1]
    hs = [torch.zeros(1, 2, HT) for _ in range(L)]; outs = []
    bounds = sorted({0, x.shape[1]} | {c for mc in mcs if mc["state_reset"] for c in range(0, x.shape[1], mc["state_reset"])})
    for a, b in zip(bounds[:-1], bounds[1:]):
        for mc, o, H in zip(mcs, offs, Hs):
            if a and mc["state_reset"] and a % mc["state_reset"] == 0:
                hs = [h.clone() for h in hs]
                for h in hs: h[..., int(o):int(o) + H] = 0
        p, hs = fused(x[:, a:b], d[:, a:b], hs); outs.append(p)
    err = (torch.cat(outs, 1) - ref).abs().max().item()
print(f"parity max abs err {err:.2e} on 2 x 2500 held-out rows")
assert err < 1e-4, err

os.makedirs(out, exist_ok=True)
spec = [dict(name=f"hidden_{i}", shape=[1, 1, HT], init="zero", reset=False,
             reset_slices=[[mc["state_reset"], int(o), int(o) + H] for mc, o, H in zip(mcs, offs, Hs) if mc["state_reset"]])
        for i in range(L)]
if any_diff:
    spec.append(dict(name="prev_x", shape=[1, 1, NF], init="x", tile=1, reset=False))
names = [s["name"] for s in spec]
dummy = (torch.zeros(1, 1, NF), *[torch.zeros(*s["shape"]) for s in spec])
torch.onnx.export(Step(fused), dummy, f"{out}/model.onnx", input_names=["features", *names],
                  output_names=["prediction", *[f"next_{n}" for n in names]], opset_version=17, dynamo=False)
json.dump(dict(hidden=HT, layers=L, state_reset=0, states=spec, members=paths, member_cfg=mcs,
               member_best=[c["state"]["best"] for c in cks]), open(f"{out}/meta.json", "w"), indent=1)
shutil.copy(os.path.join(os.path.dirname(__file__), "solution.py"), out)
print("fused", " + ".join(paths), "->", out, "hidden", HT)
