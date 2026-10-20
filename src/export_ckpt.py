"""Export a train_gru.py checkpoint to <dir>/model.onnx + meta.json + solution.py."""
import os, shutil, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Net, model_cfg, export  # noqa: E402

ck_path, out = sys.argv[1], sys.argv[2]
ck = torch.load(ck_path, map_location="cpu", weights_only=False)
mc = model_cfg(ck["cfg"])
net = Net(mc); net.load_state_dict(ck["model"])
os.makedirs(out, exist_ok=True)
export(net, mc, out, dict(best_wp=ck["state"]["best"], source=ck_path, cfg=ck["cfg"]))
shutil.copy(os.path.join(os.path.dirname(__file__), "solution.py"), out)
print("exported", ck_path, "->", out, "best", ck["state"]["best"])
