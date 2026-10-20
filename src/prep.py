"""Convert parquet sequences to float16 memmaps: X (N, 20000, 112), Y (N, 20000, 2), plus mask (N, 20000) for valid.

Resumable: rows already written are skipped (progress kept in <out>/<name>_done.npy).
"""
import argparse, os, sys, time
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wnn_connectome_starterpack"))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--parquet", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=0, help="number of sequences (0 = all)")
ap.add_argument("--stride", type=int, default=1, help="take every k-th row group")
a = ap.parse_args()

pf = pq.ParquetFile(a.parquet)
groups = list(range(0, pf.num_row_groups, a.stride))
if a.n:
    groups = groups[: a.n]
N = len(groups)
os.makedirs(a.out, exist_ok=True)
has_mask = "is_scored" in pf.schema_arrow.names
mode = "r+" if os.path.exists(f"{a.out}/{a.name}_X.npy") else "w+"
X = np.lib.format.open_memmap(f"{a.out}/{a.name}_X.npy", mode=mode, dtype=np.float16, shape=(N, SEQUENCE_LENGTH, 112))
Y = np.lib.format.open_memmap(f"{a.out}/{a.name}_Y.npy", mode=mode, dtype=np.float32, shape=(N, SEQUENCE_LENGTH, 2))
M = np.lib.format.open_memmap(f"{a.out}/{a.name}_M.npy", mode=mode, dtype=np.bool_, shape=(N, SEQUENCE_LENGTH)) if has_mask else None
done_path = f"{a.out}/{a.name}_done.npy"
done = np.load(done_path) if os.path.exists(done_path) else np.zeros(N, bool)
print(f"{a.name}: {N} sequences, {int(done.sum())} already done, mask={has_mask}", flush=True)
np.save(f"{a.out}/{a.name}_groups.npy", np.array(groups))
t0 = time.time()
for i, g in enumerate(groups):
    if done[i]:
        continue
    t = pf.read_row_group(g, use_threads=True)
    X[i] = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float16)
    Y[i] = np.column_stack([t["t0"].to_numpy(), t["t1"].to_numpy()])
    if has_mask:
        M[i] = t["is_scored"].to_numpy()
    done[i] = True
    if i % 100 == 99 or i == N - 1:
        X.flush(); Y.flush(); np.save(done_path, done)
        print(f"{i + 1}/{N} {time.time() - t0:.0f}s", flush=True)
print("PREP DONE", a.name, flush=True)
