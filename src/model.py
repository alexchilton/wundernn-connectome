"""Model, input features and ONNX export, shared by train_gru.py, export_ckpt.py and the parity checks.

Input features per row (all causal, computed from the raw 112 features):
  x                          always
  x - x_prev                 --diff        (x_prev = x at step 0, so the first difference is 0)
  x - ema_T(x), T in --ema   --ema 10,100  (ema_T(t) = ema_T(t-1) + (x_t - ema_T(t-1)) / T, started at x_0)
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
    return dict(hidden=c["hidden"], layers=c["layers"], diff=bool(c.get("diff", False)), ema=list(ema),
                proj=int(c.get("proj", 0) or 0), skip=bool(c.get("skip", False)), state_reset=int(c.get("state_reset", 0) or 0))


def feat_dim(mc):
    return NF * (1 + int(mc["diff"]) + len(mc["ema"]))


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
    return np.concatenate(parts, -1).astype(np.float32) if len(parts) > 1 else x


class Net(nn.Module):
    def __init__(self, mc):
        super().__init__()
        H, L, nin = mc["hidden"], mc["layers"], feat_dim(mc)
        self.proj = nn.Linear(nin, mc["proj"]) if mc["proj"] else None
        gin = mc["proj"] or nin
        self.grus = nn.ModuleList([nn.GRU(gin if i == 0 else H, H, batch_first=True) for i in range(L)])
        self.skip = mc["skip"]
        self.head = nn.Linear(H + (NF if self.skip else 0), 2)

    def forward(self, f, hs):
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
        self.net, self.L, self.diff = net, mc["layers"], mc["diff"]
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
        p, new = self.net(torch.cat(parts, -1) if len(parts) > 1 else x, hs)
        return (p, *new, *extra)


def state_spec(mc):
    s = [dict(name=f"hidden_{i}", shape=[1, 1, mc["hidden"]], init="zero", reset=True) for i in range(mc["layers"])]
    if mc["diff"]:
        s.append(dict(name="prev_x", shape=[1, 1, NF], init="x", tile=1, reset=False))
    if mc["ema"]:
        s.append(dict(name="ema", shape=[1, 1, NF * len(mc["ema"])], init="x", tile=len(mc["ema"]), reset=False))
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
