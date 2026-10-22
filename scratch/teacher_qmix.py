"""Per-quarter teacher: in each 5000-step quarter, greedy forward selection of members (plain mean of the selected),
fitted on even held-out sequences and scored on odd, and the reverse; compared with the global greedy mix and the
all-member mean. Uses the prediction cache out/preds/ written by teacher_mix.py."""
import sys, os, glob, numpy as np
sys.path.insert(0, "wnn_connectome_starterpack")
from utils import weighted_pearson
N = 192
Y = np.asarray(np.load("out/data/valid_Y.npy", mmap_mode="r")[:N]); M = np.array(np.load("out/data/valid_M.npy", mmap_mode="r")[:N]); M[:, :99] = False
P = {os.path.basename(f)[:-4]: np.load(f) for f in sorted(glob.glob("out/preds/*.npy"))}
C = list(P)
Q = [slice(q * 5000, (q + 1) * 5000) for q in range(4)]

def wp_pred(rows, pred):
    m = M[rows]
    return np.mean([weighted_pearson(Y[rows][m][:, k], pred[m][:, k]) for k in range(2)])

def greedy(score):
    sel, best = [], -1
    while True:
        cand = [(score(sel + [c]), c) for c in C if c not in sel]
        if not cand: break
        s, c = max(cand)
        if s <= best + 1e-5: break
        sel.append(c); best = s
    return sel

def mix(rows, sels):        # sels: one member list per quarter
    out = np.zeros((len(rows), 20000, 2), np.float32)
    for q, sel in zip(Q, sels):
        out[:, q] = np.mean([P[c][rows][:, q] for c in sel], 0)
    return out

for fr, to, nm in [(np.arange(0, N, 2), np.arange(1, N, 2), "even->odd"), (np.arange(1, N, 2), np.arange(0, N, 2), "odd->even")]:
    g = greedy(lambda s: wp_pred(fr, mix(fr, [s] * 4)))
    base = wp_pred(to, mix(to, [g] * 4))
    qs = []
    for qi in range(4):     # greedy for quarter qi, the other quarters fixed at the global selection
        def sc(s, qi=qi):
            sels = [g] * 4; sels[qi] = s
            return wp_pred(fr, mix(fr, sels))
        qs.append(greedy(sc))
    per = wp_pred(to, mix(to, qs))
    allm = wp_pred(to, mix(to, [C] * 4))
    print(f"{nm}: all-mean {allm:.5f}  global greedy {base:.5f} ({len(g)})  per-quarter {per:.5f}  gain {per - base:+.5f}  "
          f"sizes {[len(s) for s in qs]}", flush=True)
