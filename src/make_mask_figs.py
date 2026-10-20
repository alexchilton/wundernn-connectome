"""Mask-model figures for docs/tutorial.md (section 5a), from out/mask1/best.pt on held-out valid sequences."""
import sys
import numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "wnn_connectome_starterpack")
from utils import FEATURE_COLUMNS
F = "docs/fig"; plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})
X = np.load("out/data/valid_X.npy", mmap_mode="r"); M = np.load("out/data/valid_M.npy", mmap_mode="r")

class MaskGRU(nn.Module):
    def __init__(s, h):
        super().__init__(); s.gru = nn.GRU(112, h, num_layers=2, batch_first=True); s.head = nn.Linear(h, 1)
    def forward(s, x): return s.head(s.gru(x)[0]).squeeze(-1)
m = MaskGRU(64); m.load_state_dict(torch.load("out/mask1/best.pt", map_location="cpu", weights_only=False)["model"]); m.eval()
n = 48
with torch.no_grad():
    raw = torch.sigmoid(m(torch.from_numpy(np.asarray(X[:n], np.float32)))).numpy()
pos = float(np.asarray(M[192:392, 99:]).mean()); pw = (1 - pos) / pos
o = raw / np.clip(1 - raw, 1e-6, None) / pw; cal = o / (1 + o)
y = np.asarray(M[:n]).astype(int)

# A. one held-out sequence: P(scored) against the truth
s, a, b = 3, 6000, 7500
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.fill_between(np.arange(a, b), 0, y[s, a:b], step="mid", color="crimson", alpha=.35, lw=0, label="true is_scored")
ax.plot(np.arange(a, b), cal[s, a:b], color="k", lw=.8, label="mask model P(scored), calibrated")
ax.set_ylim(0, 1.05); ax.legend(loc="upper right"); ax.set_xlabel("step")
ax.set_title(f"Held-out valid sequence {s}: the mask model sees only past and current rows and flags the bursts")
fig.tight_layout(); fig.savefig(f"{F}/10_mask_trace.png"); plt.close(fig)

# B. ROC + calibration before/after
yy = y[:, 99:].ravel(); rr = raw[:, 99:].ravel(); cc = cal[:, 99:].ravel()
order = np.argsort(-rr); t = yy[order]; tpr = np.cumsum(t) / t.sum(); fpr = np.cumsum(1 - t) / (1 - t).sum()
auc = np.trapezoid(tpr, fpr)
bins = np.linspace(0, 1, 11)
def calib(p):
    i = np.clip(np.digitize(p, bins) - 1, 0, 9); return [p[i == k].mean() if (i == k).any() else np.nan for k in range(10)], [yy[i == k].mean() if (i == k).any() else np.nan for k in range(10)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(fpr, tpr, lw=1.5); ax[0].plot([0, 1], [0, 1], "--", color="grey")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate"); ax[0].set_title(f"ROC on {n} held-out sequences: AUC {auc:.3f} (0.5 = guessing)")
for p, lab in [(rr, f"raw output (mean {rr.mean():.3f})"), (cc, f"calibrated (mean {cc.mean():.3f})")]:
    xp, yp = calib(p); ax[1].plot(xp, yp, "o-", label=lab)
ax[1].plot([0, 1], [0, 1], "--", color="grey"); ax[1].set_xlabel("predicted P(scored)"); ax[1].set_ylabel("actual scored rate")
ax[1].set_title(f"Calibration: true scored rate here is {yy.mean():.3f}"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{F}/11_mask_roc_calib.png"); plt.close(fig)

# C. which single features differ most between scored and unscored rows (standardised mean difference)
Xs = np.asarray(X[192:292, 99:], np.float32).reshape(-1, 112); Ms = np.asarray(M[192:292, 99:]).ravel()
d = (Xs[Ms].mean(0) - Xs[~Ms].mean(0)) / (Xs.std(0) + 1e-6)
top = np.argsort(-np.abs(d))[:12]
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar([FEATURE_COLUMNS[i] for i in top], d[top], color=["crimson" if v > 0 else "steelblue" for v in d[top]])
ax.set_ylabel("scored mean - unscored mean\n(in feature std units)"); plt.xticks(rotation=45)
ax.set_title("Single features that differ most on scored rows (100 valid sequences)")
fig.tight_layout(); fig.savefig(f"{F}/12_mask_features.png"); plt.close(fig)
print(f"AUC {auc:.4f} raw mean {rr.mean():.4f} cal mean {cc.mean():.4f} true {yy.mean():.4f}")
print("top features", [(FEATURE_COLUMNS[i], round(float(d[i]), 3)) for i in top])
# burst stickiness
prev = y[:, 99:-1].ravel(); nxt = y[:, 100:].ravel()
print("P(scored | prev scored) %.3f, P(scored | prev not) %.3f" % (nxt[prev == 1].mean(), nxt[prev == 0].mean()))
