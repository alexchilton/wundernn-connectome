"""Export several train_gru.py checkpoints as ONE ONNX graph whose prediction is the plain mean of the members.

One graph means one session.run per row, so the per-call overhead is paid once; only the GRU maths repeats.
Each member keeps its own state tensors, prefixed m<i>_, listed in meta.json for the generic solution.py.
Members may differ in architecture but must share state_reset (one reset clock per row in solution.py).
Usage: python src/ensemble.py OUT_DIR ckpt1 ckpt2 [...]
"""
import json, os, shutil, sys
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(__file__))
from model import Net, Step, model_cfg, state_spec, NF  # noqa: E402

out, paths = sys.argv[1], sys.argv[2:]
steps, specs, mcs = [], [], []
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    mc = model_cfg(ck["cfg"]); net = Net(mc); net.load_state_dict(ck["model"]); net.eval()
    steps.append(Step(net, mc)); specs.append(state_spec(mc)); mcs.append(mc)
assert len({mc["state_reset"] for mc in mcs}) == 1, "members must share state_reset"


class Ensemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = nn.ModuleList(steps)
        self.n = [len(s) for s in specs]

    def forward(self, x, *states):
        preds, new, k = [], [], 0
        for st, n in zip(self.steps, self.n):
            o = st(x, *states[k:k + n]); k += n
            preds.append(o[0]); new.extend(o[1:])
        return (torch.stack(preds).mean(0), *new)


spec = [dict(s, name=f"m{i}_{s['name']}") for i, sp in enumerate(specs) for s in sp]
names = [s["name"] for s in spec]
os.makedirs(out, exist_ok=True)
dummy = (torch.zeros(1, 1, NF), *[torch.zeros(*s["shape"]) for s in spec])
torch.onnx.export(Ensemble().eval(), dummy, f"{out}/model.onnx", input_names=["features", *names],
                  output_names=["prediction", *[f"next_{n}" for n in names]], opset_version=17, dynamo=False)
json.dump(dict(hidden=None, layers=None, state_reset=mcs[0]["state_reset"], states=spec, members=paths, models=mcs),
          open(f"{out}/meta.json", "w"), indent=1)
shutil.copy(os.path.join(os.path.dirname(__file__), "solution.py"), out)
print("ensemble of", len(paths), "->", out)
