"""Figures for docs/tutorial.md, all computed from our own data. Run from the repo root."""
import os, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "wnn_connectome_starterpack")
from utils import FEATURE_COLUMNS, weighted_pearson  # noqa: E402

F = "docs/fig"
os.makedirs(F, exist_ok=True)
X = np.load("out/data/valid_X.npy", mmap_mode="r")
Y = np.load("out/data/valid_Y.npy", mmap_mode="r")
M = np.load("out/data/valid_M.npy", mmap_mode="r")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "font.size": 10})

# 1. one sequence: a few features, both targets, scored rows
s = 0
x, y, m = np.asarray(X[s], np.float32), np.asarray(Y[s]), np.asarray(M[s])
fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
for c in ["i0_p3", "i0_v0", "a3", "a5"]:
    ax[0].plot(x[:, FEATURE_COLUMNS.index(c)], lw=.4, label=c)
ax[0].set_title("Valid sequence 0: four of the 112 input features (already scaled by the organisers)")
ax[0].legend(loc="upper right", ncol=4)
ax[1].plot(y[:, 0], lw=.4, label="t0"); ax[1].plot(y[:, 1], lw=.4, label="t1", alpha=.8)
ax[1].axhspan(-2, 2, color="green", alpha=.05)
ax[1].set_title("Targets t0, t1 (the metric clips both to the green band [-2, 2])"); ax[1].legend(loc="upper right")
ax[2].fill_between(np.arange(20000), 0, m.astype(int), step="mid", color="crimson", lw=0)
ax[2].axvspan(0, 99, color="grey", alpha=.4)
ax[2].set_title(f"is_scored: only {m.mean():.1%} of the rows in this sequence count (grey = 99 warm-up rows)")
ax[2].set_xlabel("step in sequence (0 ... 19,999)")
fig.tight_layout(); fig.savefig(f"{F}/01_sequence.png"); plt.close(fig)

# 2. zoom: scored bursts against the target
fig, ax = plt.subplots(figsize=(11, 3))
a, b = 4000, 5000
ax.plot(np.arange(a, b), y[a:b, 0], lw=.8, label="t0")
sc = np.where(m[a:b])[0] + a
ax.scatter(sc, y[sc, 0], s=10, color="crimson", zorder=3, label="scored rows")
ax.set_title("Zoom on steps 4000-5000: scored rows come in short bursts"); ax.legend()
fig.tight_layout(); fig.savefig(f"{F}/02_zoom.png"); plt.close(fig)

# 3. target autocorrelation and scored share per sequence
lags = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
ac = [np.mean([np.corrcoef(Y[k][:-l, 0], Y[k][l:, 0])[0, 1] for k in range(20)]) for l in lags]
share = np.asarray(M[:, 99:]).mean(1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].semilogx(lags, ac, "o-"); ax[0].set_xlabel("lag (steps)"); ax[0].set_ylabel("autocorrelation of t0")
ax[0].set_title("t0 is smooth: neighbours are almost the same value")
ax[1].hist(share, bins=60); ax[1].set_xlabel("share of a sequence's rows that are scored")
ax[1].set_title(f"Scored share per valid sequence (median {np.median(share):.1%})")
fig.tight_layout(); fig.savefig(f"{F}/03_autocorr_share.png"); plt.close(fig)

# 4. the metric by counterexample: same predictions, global vs per-sequence offset
rng = np.random.default_rng(0)
ya = rng.normal(0, 1, 2000); yb = rng.normal(0, 1, 2000)
pa = ya + rng.normal(0, .8, 2000); pb = yb + rng.normal(0, .8, 2000)
good = weighted_pearson(np.r_[ya, yb], np.r_[pa, pb])
shifted = weighted_pearson(np.r_[ya, yb], np.r_[pa, pb + 1.5])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for k, (pb_, t) in enumerate([(pb, f"no offset: WP = {good:.3f}"), (pb + 1.5, f"sequence B shifted +1.5: WP = {shifted:.3f}")]):
    ax[k].scatter(ya, pa, s=3, alpha=.4, label="sequence A"); ax[k].scatter(yb, pb_, s=3, alpha=.4, label="sequence B")
    ax[k].set_title(t); ax[k].set_xlabel("target"); ax[k].legend(markerscale=4)
ax[0].set_ylabel("prediction")
fig.suptitle("One global correlation over all sequences: a per-sequence offset is pure noise to the metric")
fig.tight_layout(); fig.savefig(f"{F}/04_global_offset.png"); plt.close(fig)

# 5. the |y| weighting: which rows matter
yy = np.clip(np.asarray(Y[:50, 99:, 0]).ravel(), -2, 2); mm = np.asarray(M[:50, 99:]).ravel()
w = np.abs(yy)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].hist(yy, bins=80, weights=np.ones_like(yy) / len(yy), alpha=.6, label="row share")
ax[0].hist(yy, bins=80, weights=w / w.sum(), alpha=.6, label="weight share (|y|)")
ax[0].set_title(f"Weight |y|: {np.mean(yy == 0):.0%} of rows are exactly 0 and count for nothing"); ax[0].legend()
ax[0].set_xlabel("clipped t0")
ax[1].bar(["all rows", "scored rows only"], [w.sum(), w[mm].sum()], color=["grey", "crimson"])
ax[1].set_title(f"Scored rows carry {w[mm].sum() / w.sum():.1%} of the total |y| weight")
fig.tight_layout(); fig.savefig(f"{F}/05_weights.png"); plt.close(fig)

# 6. latency budget
fig, ax = plt.subplots(figsize=(9, 3))
rows = [("budget: 60 min / 39.4M calls", 91.4, "lightgrey"),
        ("platform ~3x local (dozken's measurement)", 30, "orange"),
        ("baseline GRU 128x2, our M4 (measured)", 18.6, "steelblue")]
for i, (lab, v, c) in enumerate(rows):
    ax.barh(i, v, color=c); ax.text(v + 1, i, f"{v:.1f} us", va="center")
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.set_xlabel("microseconds per predict() call")
ax.set_title("Speed budget: the model must run one row at a time on one CPU"); ax.set_xlim(0, 110)
fig.tight_layout(); fig.savefig(f"{F}/06_latency.png"); plt.close(fig)

# 7. graphviz diagrams
dots = {
"07_architecture": r'''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3ff", fontname="Helvetica"];
 x [label="row t\n112 features", fillcolor="#fff4d6"];
 g1 [label="GRU layer 1\n112 -> 128"]; g2 [label="GRU layer 2\n128 -> 128"];
 head [label="Linear\n128 -> 2"]; out [label="prediction t\n(t0, t1)", fillcolor="#e3f6e3"];
 h1 [label="h1 (128 numbers)", shape=ellipse, fillcolor="#f3e6ff"]; h2 [label="h2 (128 numbers)", shape=ellipse, fillcolor="#f3e6ff"];
 x -> g1 -> g2 -> head -> out;
 h1 -> g1 [label="state from row t-1"]; g1 -> h1 [label="new state", style=dashed];
 h2 -> g2; g2 -> h2 [style=dashed];
 reset [label="reset to 0:\nnew sequence, or every N rows", shape=note, fillcolor="#ffe3e3"]; reset -> h1; reset -> h2; }''',
"08_training": r'''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname="Helvetica", fillcolor="#eef3ff"];
 subgraph cluster_m { label="step 1: mask model (trained on valid, which has is_scored)"; style=rounded; color=grey;
   vx [label="valid X\n1,681 seqs", fillcolor="#fff4d6"]; mg [label="small GRU 64x2\n-> P(scored)"]; vm [label="valid is_scored", fillcolor="#ffe3e3"];
   vx -> mg; vm -> mg [label="BCE loss"]; }
 subgraph cluster_l { label="step 2: label train (train has no mask)"; style=rounded; color=grey;
   tx [label="train X", fillcolor="#fff4d6"]; tp [label="P(scored) per row", fillcolor="#ffe3e3"]; tx -> tp; }
 mg -> tp [label="predict"];
 subgraph cluster_t { label="step 3: main model"; style=rounded; color=grey;
   gru [label="GRU 2 layers"]; loss [label="loss = sum w (p - y)^2\nw = |y| x P(scored)", fillcolor="#e3f6e3"]; gru -> loss; }
 tp -> loss [label="weights"]; tx -> gru;
 onnx [label="export ONNX\n(model.onnx + meta.json)", fillcolor="#f3e6ff"]; zipf [label="submission.zip\nsolution.py + model.onnx", fillcolor="#f3e6ff"];
 gru -> onnx -> zipf; }''',
"09_tbptt": r'''digraph G { rankdir=LR; node [shape=box, style="filled", fontname="Helvetica", fillcolor="#eef3ff", width=1.1];
 c1 [label="rows 0-499"]; c2 [label="rows 500-999"]; c3 [label="rows 1000-1499"]; c4 [label="rows 1500-1999"]; c5 [label="rows 2000-2499", fillcolor="#ffe3e3"]; c6 [label="...", shape=plaintext, style=""];
 c1 -> c2 [label="state carried\ngradient cut"]; c2 -> c3; c3 -> c4; c4 -> c5 [label="state reset to 0\n(--state-reset 2000)", color=red, fontcolor=red]; c5 -> c6;
 l1 [label="loss + update", shape=ellipse, fillcolor="#e3f6e3"]; c1 -> l1 [style=dashed]; }''',
}
for name, src in dots.items():
    subprocess.run(["dot", "-Tpng", "-Gdpi=110", "-o", f"{F}/{name}.png"], input=src.encode(), check=True)
print("figures written to", F, sorted(os.listdir(F)))
