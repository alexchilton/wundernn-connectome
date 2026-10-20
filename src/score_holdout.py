"""Global WP of a solution.py directory on valid sequences [0, n) (the held-out set), step by step."""
import importlib.util, sys, time
import numpy as np
sys.path.insert(0, "wnn_connectome_starterpack")
from utils import DataPoint, weighted_pearson
d, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 192
spec = importlib.util.spec_from_file_location("sol", f"{d}/solution.py"); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
X = np.load("out/data/valid_X.npy", mmap_mode="r"); Y = np.load("out/data/valid_Y.npy", mmap_mode="r"); M = np.load("out/data/valid_M.npy", mmap_mode="r")
m = sol.PredictionModel(); ys, ps = [], []; t0 = time.time()
for s in range(n):
    x = np.asarray(X[s], dtype=np.float32); p = np.zeros((20000, 2), np.float32)
    for k in range(20000):
        r = m.predict(DataPoint(s, k, k >= 99, x[k]))
        if r is not None: p[k] = r
    mk = np.array(M[s]); mk[:99] = False
    ys.append(np.asarray(Y[s])[mk]); ps.append(p[mk])
y, p = np.concatenate(ys), np.concatenate(ps)
wp = [weighted_pearson(y[:, k], p[:, k]) for k in range(2)]
print(f"{d}: holdout{n} WP {np.mean(wp):.5f} (t0 {wp[0]:.5f} t1 {wp[1]:.5f}) {(time.time() - t0) / n / 20000 * 1e6:.1f} us/call")
