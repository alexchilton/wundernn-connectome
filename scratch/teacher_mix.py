"""Held-out WP of candidate teacher ensembles (plain mean of member predictions), valid[0:192].
Greedy forward selection fitted on even sequences, scored on odd, and the reverse; plus the all-member mean."""
import sys, os, numpy as np
sys.path.insert(0, "scratch"); sys.argv = sys.argv[:1] + ["x", "x"]
import importlib.util
spec = importlib.util.spec_from_file_location("pb", "scratch/pos_blend.py")
src = open("scratch/pos_blend.py").read().split("A, B = preds")[0]          # reuse preds() and data loading only
ns = {}; exec(compile(src, "pos_blend_head", "exec"), ns)
preds, Y, M, N = ns["preds"], ns["Y"], ns["M"], ns["N"]
from utils import weighted_pearson
C = ["out/c1000_lr1e3_e8/best.pt", "out/final_s1/best.pt", "out/final_s2/best.pt", "out/c1000_lr1e3_e8_s1/best.pt",
     "out/c2000_lr1e3_e8/best.pt", "out/c4000_lr1e3_s1/best.pt", "out/dz128_s0/best.pt", "out/dz128_s1/best.pt",
     "out/dz192_s0/best.pt", "out/dz192_s1/best.pt", "out/dz96p32_s0/latest.pt", "out/dz96p32_s1/latest.pt", "out/dzish128/best.pt"]
C = [c for c in C if os.path.exists(c)]
P = {}
for c in C:
    P[c] = preds(c); print("preds", c, flush=True)
def wp(rows, members):
    p = np.mean([P[c][rows] for c in members], 0); m = M[rows]
    return np.mean([weighted_pearson(Y[rows][m][:, k], p[m][:, k]) for k in range(2)])
allr = np.arange(N)
for c in C: print(f"single {wp(allr, [c]):.5f}  {c}")
print(f"mean of all {len(C)}: {wp(allr, C):.5f}")
print(f"e8 + final_s1 (board pair): {wp(allr, C[:2]):.5f}")
for fr, to, nm in [(np.arange(0, N, 2), np.arange(1, N, 2), "even->odd"), (np.arange(1, N, 2), np.arange(0, N, 2), "odd->even")]:
    sel, best = [], -1
    while True:
        cand = [(wp(fr, sel + [c]), c) for c in C if c not in sel]
        if not cand: break
        s, c = max(cand)
        if s <= best + 1e-5: break
        sel.append(c); best = s
    print(f"greedy {nm}: {len(sel)} members, fit {best:.5f}, held {wp(to, sel):.5f} vs board pair {wp(to, C[:2]):.5f}: {[os.path.dirname(c)[4:] for c in sel]}", flush=True)
