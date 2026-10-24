"""Checkpoint averaging (SWA over epochs) vs single epochs, held-out valid[0:192], CPU. External idea 2026-10-24."""
import sys, time, numpy as np, torch
sys.path.insert(0, "src"); sys.path.insert(0, "wnn_connectome_starterpack")
from model import Net, model_cfg, make_feats
from utils import weighted_pearson
torch.set_num_threads(8)
X = np.load("out/data/valid_X.npy", mmap_mode="r"); Y = np.load("out/data/valid_Y.npy", mmap_mode="r"); M = np.load("out/data/valid_M.npy", mmap_mode="r")
N, WARM = 192, 99
def score(sd, mc):
    net = Net(mc); net.load_state_dict(sd); net.eval(); ys, ps = [], []
    with torch.no_grad():
        for s in range(0, N, 64):
            x = torch.from_numpy(make_feats(np.asarray(X[s:s + 64], dtype=np.float32), mc))
            hs = [torch.zeros(1, x.shape[0], mc["hidden"]) for _ in range(mc["layers"])]; out = []
            for c in range(0, 20000, 2000):
                p, hs = net(x[:, c:c + 2000], hs); out.append(p.float().numpy())
            p = np.concatenate(out, 1); y = np.asarray(Y[s:s + 64]); m = np.asarray(M[s:s + 64]).copy(); m[:, :WARM] = False
            ys.append(y[m]); ps.append(p[m])
    y, p = np.concatenate(ys), np.concatenate(ps)
    return float(np.mean([weighted_pearson(y[:, k], p[:, k]) for k in range(2)]))
def avg(sds):
    return {k: sum(sd[k].float() for sd in sds) / len(sds) if sds[0][k].is_floating_point() else sds[0][k] for k in sds[0]}
for run in sys.argv[1:]:
    cks = {e: torch.load(f"out/{run}/ep{e}.pt", map_location="cpu", weights_only=False) for e in range(1, 9)}
    mc = model_cfg(cks[8]["cfg"]); t = time.time()
    for key in ("model", "raw_model"):     # model = EMA 0.999 weights when --ema-weights, raw = optimiser weights
        for e in range(3, 9):
            print(f"{run} {key} ep{e}: {score(cks[e][key], mc):.5f}", flush=True)
        for es in ((4, 5, 6), (3, 4, 5, 6), (5, 6, 7, 8), (3, 4, 5, 6, 7, 8), (6, 7, 8)):
            print(f"{run} {key} avg{es}: {score(avg([cks[e][key] for e in es]), mc):.5f}", flush=True)
    print(f"{run} done {time.time() - t:.0f}s", flush=True)
