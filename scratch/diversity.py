"""Residual correlation with a reference member and the 2-member mean, held-out valid[0:192].
usage: diversity.py REF.pt A.pt [B.pt ...]  (prints, for each candidate: single WP, residual corr with REF, mean-with-REF WP)"""
import sys
args, sys.argv = sys.argv, sys.argv[:1]          # pos_blend reads sys.argv[3] at import
ns = {}; exec(compile(open("scratch/pos_blend.py").read().split("A, B = preds")[0], "pb", "exec"), ns)
sys.argv = args
import numpy as np
from utils import weighted_pearson
preds, Y, M = ns["preds"], ns["Y"], ns["M"]
rows = np.arange(ns["N"])
def wp(P): return np.mean([weighted_pearson(Y[M][:, k], P[M][:, k]) for k in range(2)])
R = preds(sys.argv[1]); w = np.abs(np.clip(Y, -2, 2))[M]
print(f"ref {sys.argv[1]} single {wp(R):.5f}")
for c in sys.argv[2:]:
    A = preds(c)
    rc = [np.corrcoef((A[M][:, k] - Y[M][:, k]) * np.sqrt(w[:, k]), (R[M][:, k] - Y[M][:, k]) * np.sqrt(w[:, k]))[0, 1] for k in range(2)]
    print(f"{c}: single {wp(A):.5f}  weighted residual corr with ref {np.mean(rc):.4f}  mean with ref {wp((A + R) / 2):.5f}", flush=True)
