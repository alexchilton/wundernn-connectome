"""Global WP per sequence quarter on held-out valid[0:n] for a checkpoint (batched torch, CPU)."""
import sys, numpy as np, torch
sys.path.insert(0, "wnn_connectome_starterpack"); sys.path.insert(0, "src")
from utils import weighted_pearson
from model import Net, model_cfg, make_feats
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); mc = model_cfg(ck["cfg"])
net = Net(mc); net.load_state_dict(ck["model"]); net.eval()
n = int(sys.argv[2]) if len(sys.argv) > 2 else 192
X = np.load("out/data/valid_X.npy", mmap_mode="r"); Y = np.load("out/data/valid_Y.npy", mmap_mode="r"); M = np.load("out/data/valid_M.npy", mmap_mode="r")
P = np.zeros((n, 20000, 2), np.float32)
with torch.no_grad():
    for s in range(0, n, 32):
        f = torch.from_numpy(make_feats(np.asarray(X[s:s + 32], np.float32), mc))
        hs = [torch.zeros(1, f.shape[0], mc["hidden"]) for _ in range(mc["layers"])]
        step = min(2000, mc["state_reset"]) if mc["state_reset"] else 2000
        for c in range(0, 20000, step):
            if mc["state_reset"] and c % mc["state_reset"] == 0:
                hs = [torch.zeros_like(h) for h in hs]
            p, hs = net(f[:, c:c + step], hs); P[s:s + 32, c:c + step] = p.numpy()
y = np.asarray(Y[:n]); m = np.array(M[:n]); m[:, :99] = False
def wp(sl):
    mm = np.zeros_like(m); mm[:, sl] = m[:, sl]
    return np.mean([weighted_pearson(y[mm][:, k], P[mm][:, k]) for k in range(2)])
print(sys.argv[1], "full %.4f" % wp(slice(0, 20000)), "| quarters", " ".join("%.4f" % wp(slice(q * 5000, (q + 1) * 5000)) for q in range(4)))
