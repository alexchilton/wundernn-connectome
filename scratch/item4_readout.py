"""Fable plan 2026-10-23 item 4 read-out: score e8th_s0's 8 epoch checkpoints (EMA weights, 'model' key) on
  valid[0:192] with the true mask (the usual held-out WP), and
  train[:192] (held out of this run's pool by --train-holdout 192) with weights |y| * P(scored), and with plain |y|.
Pre-registered: same peak epoch and Spearman >= 0.8 between the two rankings kills the regime story locally.
    .venv/bin/python -u scratch/item4_readout.py > logs/item4_readout.log"""
import sys, numpy as np, torch
sys.path.insert(0, "src")
from model import Net, model_cfg, make_feats
WARMUP, L = 99, 20000
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def wp(y, p, w):
    y, p = np.clip(y, -2, 2).astype(np.float64), np.clip(p, -2, 2).astype(np.float64)
    W = w.sum(); my, mp = (w * y).sum() / W, (w * p).sum() / W
    return float((w * (y - my) * (p - mp)).sum() / np.sqrt((w * (y - my) ** 2).sum() * (w * (p - mp) ** 2).sum()))

def predict(net, mc, X):
    out = []
    for s in range(0, X.shape[0], 32):
        x = torch.from_numpy(make_feats(np.asarray(X[s:s + 32], dtype=np.float32), mc)).to(dev)
        hs = [torch.zeros(1, x.shape[0], mc["hidden"], device=dev) for _ in range(mc["layers"])]
        ps = []
        with torch.no_grad():
            for c in range(0, L, 2000):
                p, hs = net(x[:, c:c + 2000], hs); ps.append(p.float().cpu().numpy())
        out.append(np.concatenate(ps, 1))
    return np.concatenate(out)

Xv, Yv, Mv = (np.load(f"out/data/valid_{k}.npy", mmap_mode="r")[:192] for k in "XYM")
Xt, Yt, Pt = (np.load(f"out/data/train192_{k}.npy", mmap_mode="r") for k in "XYP")
rows = []
for ep in range(1, 9):
    ck = torch.load(f"out/e8th_s0/ep{ep}.pt", map_location="cpu", weights_only=False)
    mc = model_cfg(ck["cfg"]); net = Net(mc).to(dev); net.load_state_dict(ck["model"]); net.eval()
    pv = predict(net, mc, Xv); yv = np.asarray(Yv, np.float32); m = np.asarray(Mv).copy(); m[:, :WARMUP] = False
    v = np.mean([wp(yv[..., k][m], pv[..., k][m], np.abs(np.clip(yv[..., k][m], -2, 2))) for k in range(2)])
    pt = predict(net, mc, Xt); yt = np.asarray(Yt, np.float32); P = np.asarray(Pt, np.float32)
    sl = np.s_[:, WARMUP:]
    tP = np.mean([wp(yt[sl][..., k].ravel(), pt[sl][..., k].ravel(), (np.abs(np.clip(yt[sl][..., k], -2, 2)) * P[sl]).ravel()) for k in range(2)])
    tA = np.mean([wp(yt[sl][..., k].ravel(), pt[sl][..., k].ravel(), np.abs(np.clip(yt[sl][..., k], -2, 2)).ravel()) for k in range(2)])
    rows.append((ep, v, tP, tA)); print(f"ep{ep}: valid[0:192] {v:.5f}  train[:192] |y|P {tP:.5f}  |y| {tA:.5f}", flush=True)
from scipy.stats import spearmanr
r = np.array(rows)
print("peak epoch: valid", int(r[np.argmax(r[:, 1]), 0]), "| train |y|P", int(r[np.argmax(r[:, 2]), 0]), "| train |y|", int(r[np.argmax(r[:, 3]), 0]))
print("Spearman valid vs train|y|P", round(spearmanr(r[:, 1], r[:, 2]).correlation, 3), "| vs train|y|", round(spearmanr(r[:, 1], r[:, 3]).correlation, 3))
