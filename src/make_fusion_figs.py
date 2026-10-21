"""Figures 13-16 for docs/tutorial.md (chunk vs reset, fusion, timing). Run from the repo root after src/fuse.py has
written out/fz_e8_s1. Every number drawn is measured; the sources are listed in the tutorial next to each figure."""
import os
import numpy as np
import onnx
from onnx import numpy_helper
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F = "docs/fig"
os.makedirs(F, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})

# 13. chunk vs reset: what happens to the memory and to the gradient along one sequence
fig, ax = plt.subplots(2, 1, figsize=(11, 5.2), sharex=True)
T = 8000
t = np.arange(T)
for a, (title, chunk, reset) in zip(ax, [("chunk 1000, no reset: memory runs on through all rows; gradients stop at each chunk edge", 1000, 0),
                                          ("chunk 1000, reset every 2000: memory is wiped to zero every 2000 rows, in training AND at inference", 1000, 2000)]):
    mem = np.zeros(T)
    h = 0.0
    for i in range(T):
        if reset and i % reset == 0:
            h = 0.0
        h = 0.995 * h + 0.005 * (1 + 0.3 * np.sin(i / 300))
        mem[i] = h
    a.plot(t, mem, color="tab:blue", lw=1.5, label="how much the memory holds (sketch)")
    for c in range(0, T, chunk):
        a.axvline(c, color="tab:orange", ls="--", lw=1)
    if reset:
        for r in range(0, T, reset):
            a.axvline(r, color="crimson", lw=2.5, alpha=.7)
    a.set_title(title, fontsize=10)
    a.set_ylim(0, 1.25); a.set_yticks([])
ax[0].plot([], [], color="tab:orange", ls="--", label="chunk edge: gradient cut (training only)")
ax[1].plot([], [], color="crimson", lw=2.5, label="reset: memory set to zero")
ax[0].legend(loc="lower right", fontsize=8); ax[1].legend(loc="lower right", fontsize=8)
ax[1].set_xlabel("row in the sequence (first 8,000 of 20,000)")
fig.tight_layout(); fig.savefig(f"{F}/13_chunk_vs_reset.png"); plt.close(fig)

# 14. the fused recurrent weight matrix, read from the real exported model: two blocks, zeros elsewhere
m = onnx.load("out/fz_e8_s1/model.onnx")
W = None
for init in m.graph.initializer:
    a = numpy_helper.to_array(init)
    if a.ndim == 3 and a.shape[1] == 3 * 256 and a.shape[2] == 256:   # ONNX GRU recurrence weight R: (1, 3H, H)
        W = a[0]
        break
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
im = ax[0].imshow(np.abs(W[:256]) > 1e-12, cmap="Greys", aspect="auto", interpolation="nearest")
ax[0].set_title("Fused GRU, one gate's recurrent weights (256 x 256)\nblack = non-zero, white = exactly zero")
ax[0].set_xlabel("memory unit read (0-127 model A, 128-255 model B)"); ax[0].set_ylabel("memory unit written")
ax[0].grid(False)
ax[1].axis("off")
ax[1].text(0.0, 0.95, "Why the output is exactly the average of the two models", fontsize=11, weight="bold", va="top")
ax[1].text(0.0, 0.83, "Top-left block   = model A's weights\nBottom-right block = model B's weights\n"
           "Off-diagonal blocks = 0:\n   A's units never read B's units and vice versa.\n\n"
           "So the wide GRU is two GRUs side by side in one op.\n"
           "The head reads A's half and B's half with each\nmodel's own head and averages the two answers.\n\n"
           "Checked: fused output vs the two models run\nseparately and averaged: max difference 1.1e-6.",
           fontsize=10, va="top", family="monospace")
fig.tight_layout(); fig.savefig(f"{F}/14_fused_matrix.png"); plt.close(fig)

# 15. what averaging bought on held-out and on the public board (measured)
labels = ["full1\n(single)", "seed 1\n(single)", "e8\n(single)", "s1 + s2\nfused", "s1 + reset2k\nfused", "e8 + s1\nfused"]
held = [0.68310, 0.68488, 0.68703, 0.68588, 0.68655, 0.68858]
pub = [0.65392, None, None, 0.65625, 0.65599, None]
fig, ax = plt.subplots(figsize=(11, 3.8))
x = np.arange(len(labels))
ax.bar(x, held, color=["tab:grey"] * 3 + ["tab:blue"] * 3)
for i, (h, p) in enumerate(zip(held, pub)):
    ax.text(i, h + 0.0003, f"{h:.5f}" + (f"\npublic {p:.5f}" if p else ""), ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0.680, 0.692); ax.set_ylabel("held-out WP (valid 0-191)")
ax.set_title("Two different models averaged beat either one; the gain carries to the public board")
fig.tight_layout(); fig.savefig(f"{F}/15_fusion_gain.png"); plt.close(fig)

# 16. platform wall time against model width (measured uploads) and the 60-minute limit
fig, ax = plt.subplots(figsize=(8, 3.6))
names = ["baseline 128x2", "full1 128x2", "fused 256 (s1+s2)", "fused 256 (s1+r2k)\nIOBinding loop"]
mins = [26 + 36 / 60, 26 + 12 / 60, 46 + 12 / 60, 47 + 6 / 60]
ax.barh(names, mins, color=["tab:grey", "tab:grey", "tab:blue", "tab:blue"])
ax.axvline(60, color="crimson", lw=2); ax.text(60.5, 3.3, "60 min limit", color="crimson")
for i, v in enumerate(mins):
    ax.text(v + 0.5, i, f"{v:.1f} min", va="center")
ax.set_xlim(0, 70); ax.set_xlabel("platform wall time for 39.4M rows (minutes)")
ax.set_title("Two fused 128s fit with ~13 min to spare; three (384 wide) would not")
fig.tight_layout(); fig.savefig(f"{F}/16_platform_time.png"); plt.close(fig)
print("wrote figures 13-16")
