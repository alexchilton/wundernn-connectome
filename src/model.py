"""Model, input features and ONNX export, shared by train_gru.py, export_ckpt.py and the parity checks.

Input features per row (all causal, computed from the raw 112 features):
  x                          always
  x - x_prev                 --diff        (x_prev = x at step 0, so the first difference is 0)
  x - ema_T(x), T in --ema   --ema 10,100  (ema_T(t) = ema_T(t-1) + (x_t - ema_T(t-1)) / T, started at x_0)
  log(1 + step) / log(20000) --time        (position in the sequence; carried as a step counter state)
Network: optional Linear projection (--proj), stacked single-layer GRUs, linear head on the top GRU output,
optionally concatenated with the raw row (--skip).

At inference the ONNX step graph carries every piece of memory as an explicit state tensor:
hidden_k for each GRU layer (zeroed at a new sequence and on --state-reset), prev_x and ema (set from the
first row of a sequence, never reset). meta.json lists them, so solution.py needs no model-specific code.
With no options set, the parameter names match the original StackedGRU, so older checkpoints load.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import lfilter

NF = 112


def model_cfg(c):
    """The architecture keys of a training config (argparse vars or a saved cfg dict)."""
    ema = c.get("ema") or []
    if isinstance(ema, str):
        ema = [float(t) for t in ema.split(",") if t]
    mc = dict(hidden=c["hidden"], layers=c["layers"], diff=bool(c.get("diff", False)), ema=list(ema),
              proj=int(c.get("proj", 0) or 0), skip=bool(c.get("skip", False)), time=bool(c.get("time", False)), state_reset=int(c.get("state_reset", 0) or 0))
    if c.get("dz"):     # only present when set, so configs of older checkpoints compare equal
        mc.update(dz=True, dropout=float(c.get("dropout", 0.1)))
    if c.get("aux"):    # auxiliary training targets: extra head rows, never exported
        mc.update(aux=int(c["aux"]))
    if c.get("bidir"):  # teacher only (train_gru.py --bidir), never exported
        mc.update(bidir=True)
    if c.get("tf"):
        mc.update(tf=True, window=int(c.get("window", 256)), heads=int(c.get("heads", 4)), dropout=float(c.get("dropout", 0.1)))
    return mc


def feat_dim(mc):
    return NF * (1 + int(mc["diff"]) + len(mc["ema"])) + int(mc.get("time", False))


def make_feats(x, mc):
    """x: (B, T, 112) float32, whole sequences from step 0. Returns (B, T, feat_dim) float32."""
    parts = [x]
    if mc["diff"]:
        d = np.zeros_like(x)
        d[:, 1:] = x[:, 1:] - x[:, :-1]
        parts.append(d)
    for T in mc["ema"]:
        al = 1.0 / T
        e, _ = lfilter([al], [1.0, -(1.0 - al)], x, axis=1, zi=(1.0 - al) * x[:, :1, :])
        parts.append(x - e)
    if mc.get("time"):
        t = (np.log1p(np.arange(x.shape[1], dtype=np.float32)) / np.float32(np.log(20000.0)))[None, :, None]
        parts.append(np.broadcast_to(t, (x.shape[0], x.shape[1], 1)))
    return np.concatenate(parts, -1).astype(np.float32) if len(parts) > 1 else x


class TFBlock(nn.Module):
    """Pre-LN causal self-attention over [cache of the previous W layer inputs, this chunk], ALiBi distance bias,
    then a 4x GELU MLP."""

    def __init__(self, d, heads, drop):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, cache, bias):
        B, C, d = x.shape
        u = torch.cat([cache, x], 1) if cache is not None else x
        q = self.qkv(self.ln1(x))[..., :d]
        kv = self.qkv(self.ln1(u))[..., d:]
        k, v = kv[..., :d], kv[..., d:]
        sh = lambda t: t.view(B, t.shape[1], self.h, d // self.h).transpose(1, 2)
        a = nn.functional.scaled_dot_product_attention(sh(q), sh(k), sh(v), attn_mask=bias)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, C, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class TFNet(nn.Module):
    """Causal local-window transformer (teacher / research model, not exported). The state per layer is the last
    W inputs of that layer; a state that is not (B, W, d) (the GRU-shaped zeros the trainers start with) means
    'sequence start, empty cache'."""

    def __init__(self, mc):
        super().__init__()
        d, L, self.W, self.h = mc["hidden"], mc["layers"], mc["window"], mc["heads"]
        self.diff = mc["diff"]
        self.register_buffer("mean", torch.zeros(NF)); self.register_buffer("std", torch.ones(NF))
        self.inp = nn.Linear(feat_dim(mc), d)
        self.blocks = nn.ModuleList([TFBlock(d, self.h, mc.get("dropout", 0.1)) for _ in range(L)])
        self.lnf, self.head = nn.LayerNorm(d), nn.Linear(d, 2)
        self.register_buffer("slopes", torch.tensor([2.0 ** (-8.0 * (i + 1) / self.h) for i in range(self.h)]))

    def bias(self, C, P, dev):
        i = torch.arange(C, device=dev)[:, None] + P                 # query time, P = cached positions before it
        j = torch.arange(C + P, device=dev)[None, :]
        dist = (i - j).float()
        ok = (dist >= 0) & (dist <= self.W)
        b = -self.slopes[:, None, None] * dist[None]
        return b.masked_fill(~ok[None], float("-inf"))[None]           # (1, heads, C, C + P)

    def forward(self, f, hs):
        x = ((f[..., :NF] - self.mean) / self.std).clamp(-10.0, 10.0)
        if self.diff:
            x = torch.cat([x, f[..., NF:2 * NF] / self.std], -1)
        z = self.inp(torch.cat([x, f[..., x.shape[-1]:]], -1) if f.shape[-1] > x.shape[-1] else x)
        B, C, _ = z.shape
        caches = [h if (h.dim() == 3 and h.shape[0] == B and h.shape[1] == self.W) else None for h in hs]
        P = self.W if caches[0] is not None else 0
        bias = self.bias(C, P, z.device).to(z.dtype)
        new = []
        for blk, c in zip(self.blocks, caches):
            u = torch.cat([c, z], 1) if c is not None else z
            assert u.shape[1] >= self.W, "chunk must be >= window at a sequence start"
            new.append(u[:, -self.W:])
            z = blk(z, c, bias)
        return self.head(self.lnf(z)), new


class BiNet(nn.Module):
    """Teacher only, never exported (findings 5: architectural diversity for distillation): a forward and a backward GRU
    stack over the same plain features, linear head on [forward, backward]. Every row sees the whole sequence.
    Truncated at chunk edges in both directions: bwd_states() is a no-grad right-to-left pass giving the backward state
    entering each chunk from the right; chunk() runs one chunk with the forward state carried from the left."""

    def __init__(self, mc):
        super().__init__()
        H, L, nin = mc["hidden"], mc["layers"], feat_dim(mc)
        assert not (mc["proj"] or mc["skip"] or mc.get("dz") or mc.get("aux")), "bidir: plain features only"
        self.f = nn.ModuleList([nn.GRU(nin if i == 0 else H, H, batch_first=True) for i in range(L)])
        self.b = nn.ModuleList([nn.GRU(nin if i == 0 else H, H, batch_first=True) for i in range(L)])
        self.head = nn.Linear(2 * H, 2)

    @staticmethod
    def _run(grus, z, hs):
        new = []
        for g, h in zip(grus, hs):
            z, h = g(z, h)
            new.append(h)
        return z, new

    @torch.no_grad()
    def bwd_states(self, f, step):
        hs = [f.new_zeros(1, f.shape[0], g.hidden_size) for g in self.b]
        st = {}
        for c in reversed(range(0, f.shape[1], step)):
            st[c] = hs
            _, hs = self._run(self.b, f[:, c:c + step].flip(1), hs)
        return st

    def chunk(self, fc, hf, hb):
        zf, hf = self._run(self.f, fc, hf)
        zb, _ = self._run(self.b, fc.flip(1), hb)
        return self.head(torch.cat([zf, zb.flip(1)], -1)), hf


class Net(nn.Module):
    """Plain: optional Linear projection, stacked GRUs, linear head (optionally also on the raw row).
    dz=True (the block design read in dozken/wunder challenge-3/src/model.py, our own implementation): raw row
    standardised by train mean/std and clipped at +-10 (diff scaled by the same std), projection Linear -> GELU ->
    LayerNorm, dropout between GRU layers, head LayerNorm(concat(GRU out, projected input)) -> Linear(H) -> GELU ->
    Dropout -> Linear(2)."""

    def __new__(cls, mc=None):                     # mc is None when copy.deepcopy rebuilds a Net (EMA AveragedModel)
        if mc is not None and mc.get("bidir"):
            return BiNet(mc)
        return TFNet(mc) if mc is not None and mc.get("tf") else super().__new__(cls)

    def __init__(self, mc):
        super().__init__()
        H, L, nin = mc["hidden"], mc["layers"], feat_dim(mc)
        self.dz = bool(mc.get("dz"))
        if self.dz:
            assert mc["proj"] and not mc["skip"] and not mc["ema"] and not mc.get("time"), "dz: needs --proj, no skip/ema/time"
            self.register_buffer("mean", torch.zeros(NF)); self.register_buffer("std", torch.ones(NF))
            P, self.diff = mc["proj"], mc["diff"]
            self.inproj = nn.Sequential(nn.Linear(nin, P), nn.GELU(), nn.LayerNorm(P))
            self.grus = nn.ModuleList([nn.GRU(P if i == 0 else H, H, batch_first=True) for i in range(L)])
            self.drop = nn.Dropout(mc.get("dropout", 0.1))
            self.head_norm = nn.LayerNorm(H + P)
            self.head = nn.Sequential(nn.Linear(H + P, H), nn.GELU(), nn.Dropout(mc.get("dropout", 0.1)), nn.Linear(H, 2 + int(mc.get("aux", 0))))
            return
        self.proj = nn.Linear(nin, mc["proj"]) if mc["proj"] else None
        gin = mc["proj"] or nin
        self.grus = nn.ModuleList([nn.GRU(gin if i == 0 else H, H, batch_first=True) for i in range(L)])
        self.skip = mc["skip"]
        self.head = nn.Linear(H + (NF if self.skip else 0), 2 + int(mc.get("aux", 0)))

    def forward(self, f, hs):
        if self.dz:
            x = ((f[..., :NF] - self.mean) / self.std).clamp(-10.0, 10.0)
            u = torch.cat([x, f[..., NF:2 * NF] / self.std], -1) if self.diff else x
            z = self.inproj(u); o = z; new = []
            for i, (gru, h) in enumerate(zip(self.grus, hs)):
                o, h = gru(self.drop(o) if i else o, h)
                new.append(h)
            return self.head(self.head_norm(torch.cat([o, z], -1))), new
        z = self.proj(f) if self.proj is not None else f
        new = []
        for gru, h in zip(self.grus, hs):
            z, h = gru(z, h)
            new.append(h)
        if self.skip:
            z = torch.cat([z, f[..., :NF]], -1)
        return self.head(z), new


class Step(nn.Module):
    """One row in, one prediction out; all memory passed in and out explicitly."""

    def __init__(self, net, mc):
        super().__init__()
        self.net, self.L, self.diff, self.time = net, mc["layers"], mc["diff"], mc.get("time", False)
        self.K = len(mc["ema"])
        if self.K:
            self.register_buffer("alpha", torch.tensor(np.repeat([1.0 / T for T in mc["ema"]], NF), dtype=torch.float32))

    def forward(self, x, *states):
        hs, k, parts, extra = list(states[:self.L]), self.L, [x], []
        if self.diff:
            prev = states[k]; k += 1
            parts.append(x - prev); extra.append(x)
        if self.K:
            ema = states[k]; k += 1
            xt = x.repeat(1, 1, self.K)
            ema = ema + self.alpha * (xt - ema)
            parts.append(xt - ema); extra.append(ema)
        if self.time:
            t = states[k]; k += 1
            parts.append(torch.log1p(t) / float(np.log(20000.0))); extra.append(t + 1.0)
        p, new = self.net(torch.cat(parts, -1) if len(parts) > 1 else x, hs)
        return (p[..., :2], *new, *extra)


def state_spec(mc):
    s = [dict(name=f"hidden_{i}", shape=[1, 1, mc["hidden"]], init="zero", reset=True) for i in range(mc["layers"])]
    if mc["diff"]:
        s.append(dict(name="prev_x", shape=[1, 1, NF], init="x", tile=1, reset=False))
    if mc["ema"]:
        s.append(dict(name="ema", shape=[1, 1, NF * len(mc["ema"])], init="x", tile=len(mc["ema"]), reset=False))
    if mc.get("time"):
        s.append(dict(name="step", shape=[1, 1, 1], init="zero", reset=False))
    return s


def export(net, mc, out_dir, extra_meta=None):
    net = net.cpu().eval()
    spec = state_spec(mc)
    names = [s["name"] for s in spec]
    dummy = (torch.zeros(1, 1, NF), *[torch.zeros(*s["shape"]) for s in spec])
    torch.onnx.export(Step(net, mc), dummy, f"{out_dir}/model.onnx", input_names=["features", *names],
                      output_names=["prediction", *[f"next_{n}" for n in names]], opset_version=17, dynamo=False)
    meta = dict(hidden=mc["hidden"], layers=mc["layers"], state_reset=mc["state_reset"], states=spec, model=mc)
    meta.update(extra_meta or {})
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)
