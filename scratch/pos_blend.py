"""Position-weighted blend of two members on held-out valid[0:192]: weight w_s for member A in each step segment s
(per target), fitted on one half of the sequences (even / odd) and scored on the other. Plain mean as the control."""
import sys, os, numpy as np, torch
sys.path.insert(0, "wnn_connectome_starterpack"); sys.path.insert(0, "src")
from utils import weighted_pearson
from model import Net, model_cfg, make_feats
N, SEG = 192, int(sys.argv[3]) if len(sys.argv) > 3 else 2000
X = np.load("out/data/valid_X.npy", mmap_mode="r"); Y = np.asarray(np.load("out/data/valid_Y.npy", mmap_mode="r")[:N])
M = np.array(np.load("out/data/valid_M.npy", mmap_mode="r")[:N]); M[:, :99] = False

def preds(path):
    cache = f"out/preds/{path.replace('/', '_')}.npy"
    if os.path.exists(cache):
        return np.load(cache)
    ck = torch.load(path, map_location="cpu", weights_only=False); mc = model_cfg(ck["cfg"])
    net = Net(mc); net.load_state_dict(ck["model"]); net.eval()
    P = np.zeros((N, 20000, 2), np.float32)
    with torch.no_grad():
        for s in range(0, N, 32):
            f = torch.from_numpy(make_feats(np.asarray(X[s:s + 32], np.float32), mc))
            hs = [torch.zeros(1, f.shape[0], mc["hidden"]) for _ in range(mc["layers"])]
            step = min(2000, mc["state_reset"]) if mc["state_reset"] else 2000
            for c in range(0, 20000, step):
                if mc["state_reset"] and c % mc["state_reset"] == 0:
                    hs = [torch.zeros_like(h) for h in hs]
                p, hs = net(f[:, c:c + step], hs); P[s:s + 32, c:c + step] = p.numpy()
    os.makedirs("out/preds", exist_ok=True); np.save(cache, P)
    return P

A, B = preds(sys.argv[1]), preds(sys.argv[2])
S = 20000 // SEG
def score(rows, w):                       # w: (S, 2) weight of A per segment and target
    wt = np.repeat(w, SEG, axis=0)        # (20000, 2)
    P = wt * A[rows] + (1 - wt) * B[rows]
    m = M[rows]
    return np.mean([weighted_pearson(Y[rows][m][:, k], P[m][:, k]) for k in range(2)])

def fit(rows):
    w = np.full((S, 2), 0.5)
    best = score(rows, w)
    for _ in range(2):
        for s in range(S):
            for k in range(2):
                for v in np.linspace(0, 1, 11):
                    w2 = w.copy(); w2[s, k] = v; sc = score(rows, w2)
                    if sc > best + 1e-7:
                        best, w = sc, w2
    return w

ev, od = np.arange(0, N, 2), np.arange(1, N, 2)
half = np.full((S, 2), 0.5)
print(f"A={sys.argv[1]} B={sys.argv[2]} segment {SEG}")
print(f"single A {score(np.arange(N), np.ones((S, 2))):.5f}  single B {score(np.arange(N), np.zeros((S, 2))):.5f}  mean {score(np.arange(N), half):.5f}")
gain = []
for fr, to, nm in [(ev, od, "fit even -> score odd"), (od, ev, "fit odd -> score even")]:
    w = fit(fr)
    g = score(to, w) - score(to, half)
    gain.append(g)
    print(f"{nm}: mean {score(to, half):.5f}  pos-weighted {score(to, w):.5f}  gain {g:+.5f}  weights A t0 {np.round(w[:, 0], 1)} t1 {np.round(w[:, 1], 1)}", flush=True)
print(f"cross-fitted gain {np.mean(gain):+.5f}")
