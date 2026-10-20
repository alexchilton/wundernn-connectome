"""Export a train_gru.py checkpoint to <dir>/model.onnx + meta.json + solution.py (same graph and interface as train_gru's own export)."""
import json, shutil, sys
import torch, torch.nn as nn
ck_path, out = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location="cpu", weights_only=False); cfg = ck["cfg"]
H, L = cfg["hidden"], cfg["layers"]

class StackedGRU(nn.Module):
    def __init__(s):
        super().__init__(); s.grus = nn.ModuleList([nn.GRU(112 if i == 0 else H, H, batch_first=True) for i in range(L)]); s.head = nn.Linear(H, 2)
    def forward(s, x, *hs):
        new = []
        for g, h in zip(s.grus, hs):
            x, h = g(x, h); new.append(h)
        return (s.head(x), *new)
m = StackedGRU(); m.load_state_dict(ck["model"]); m.eval()
import os; os.makedirs(out, exist_ok=True)
names = [f"hidden_{i}" for i in range(L)]
torch.onnx.export(m, (torch.zeros(1, 1, 112), *[torch.zeros(1, 1, H) for _ in range(L)]), f"{out}/model.onnx",
                  input_names=["features", *names], output_names=["prediction", *[f"next_{n}" for n in names]], opset_version=17, dynamo=False)
json.dump(dict(hidden=H, layers=L, state_reset=cfg.get("state_reset", 0), best_wp=ck["state"]["best"], source=ck_path, cfg=cfg), open(f"{out}/meta.json", "w"), indent=1)
shutil.copy("src/solution.py", out)
print("exported", ck_path, "->", out, "best", ck["state"]["best"])
