"""Teacher predictions for --teacher-preds: the plain mean of causal checkpoints over every sequence of one dataset,
run exactly like train_gru.evaluate() (2000-row chunks, the member's own state reset). Writes <out>/<name>_T.npy (float16,
same shape as <data>/<name>_Y.npy). Resumable: sequences done so far are in <out>/<name>_T.done.npy.
    python -u src/dump_teacher.py DATA NAME OUT CKPT [CKPT ...]"""
import os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import Net, model_cfg, make_feats

data, name, out, ckpts = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
X = np.load(f"{data}/{name}_X.npy", mmap_mode="r")
os.makedirs(out, exist_ok=True)
tp, dp = f"{out}/{name}_T.npy", f"{out}/{name}_T.done.npy"
if os.path.exists(tp) and os.path.exists(dp):
    T = np.load(tp, mmap_mode="r+"); done = np.load(dp)
else:
    T = np.lib.format.open_memmap(tp, mode="w+", dtype=np.float16, shape=X.shape[:2] + (2,)); done = np.zeros(X.shape[0], bool)
print(f"{name}: {X.shape[0]} seqs, {done.sum()} done before, {len(ckpts)} members on {dev}", flush=True)
nets = []
for p in ckpts:
    ck = torch.load(p, map_location="cpu", weights_only=False); mc = model_cfg(ck["cfg"])
    n = Net(mc); n.load_state_dict(ck["model"]); nets.append((n.to(dev).eval(), mc, p))
B, t0 = int(os.environ.get("DUMP_B", 128)), time.time()
todo = np.flatnonzero(~done)[:int(os.environ.get("DUMP_N", 0)) or None]   # DUMP_N: smoke test on the first N
with torch.no_grad():
    for b in range(0, len(todo), B):
        idx = todo[b:b + B]
        raw = np.asarray(X[idx], dtype=np.float32)
        acc = np.zeros((len(idx), X.shape[1], 2), np.float64)
        for n, mc, _ in nets:
            x = torch.from_numpy(make_feats(raw, mc)).to(dev)
            hs = [torch.zeros(1, len(idx), mc["hidden"], device=dev) for _ in range(mc["layers"])]
            sr = mc.get("state_reset", 0); step = min(2000, sr) if sr else 2000
            outs = []
            bst = n.bwd_states(x, step) if mc.get("bidir") else None
            for c in range(0, X.shape[1], step):
                if sr and c % sr == 0:
                    hs = [torch.zeros_like(h) for h in hs]
                p, hs = n.chunk(x[:, c:c + step], hs, bst[c]) if mc.get("bidir") else n(x[:, c:c + step], hs)
                outs.append(p[..., :2].float().cpu().numpy())
            acc += np.concatenate(outs, 1)
        T[idx] = (acc / len(nets)).astype(np.float16); done[idx] = True
        T.flush(); np.save(dp, done)
        el = time.time() - t0
        print(f"{done.sum()}/{X.shape[0]} seqs, {el:.0f}s, eta {el / (b + len(idx)) * (len(todo) - b - len(idx)):.0f}s", flush=True)
print(f"DUMP DONE {tp}", flush=True)
