"""Predict the hidden scoring mask (is_scored) from the features, causally.

train: fit a small GRU on valid sequences [holdout:], report AUC / AP on valid [:holdout].
label: write P(scored) per row for a memmap set (<data>/<name>_P.npy, float16), for the soft-mask loss.
Checkpoint every --save-every batches, --resume restores model, optimiser, epoch, batch and RNG.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

WARMUP, L_SEQ, NF = 99, 20000, 112

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["train", "label"])
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--name", default="", help="label: memmap set to label")
ap.add_argument("--holdout", type=int, default=192)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--chunk", type=int, default=1000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--save-every", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = torch.device(a.device)
print("CONFIG", json.dumps(vars(a)), flush=True)


class MaskGRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(NF, hidden, num_layers=2, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, h=None):
        y, h = self.gru(x, h)
        return self.head(y).squeeze(-1), h


def auc_ap(y, s):
    o = np.argsort(-s); y = y[o]
    tp = np.cumsum(y); fp = np.cumsum(1 - y); P = tp[-1]; N = fp[-1]
    auc = np.trapezoid(tp / P, fp / N)
    ap_ = float(np.sum((tp / np.arange(1, len(y) + 1))[y == 1]) / P)
    return float(auc), ap_


@torch.no_grad()
def predict(model, X, lo, hi):
    out = []
    for s in range(lo, hi, 64):
        x = torch.from_numpy(np.asarray(X[s:min(s + 64, hi)], dtype=np.float32)).to(dev)
        h, ps = None, []
        for c in range(0, L_SEQ, 2000):
            p, h = model(x[:, c:c + 2000], h)
            ps.append(torch.sigmoid(p).float().cpu().numpy())
        out.append(np.concatenate(ps, 1))
    return np.concatenate(out)


model = MaskGRU(a.hidden).to(dev)
if a.cmd == "label":
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev, weights_only=False)["model"]); model.eval()
    X = np.load(f"{a.data}/{a.name}_X.npy", mmap_mode="r")
    P = np.lib.format.open_memmap(f"{a.data}/{a.name}_P.npy", mode="w+", dtype=np.float16, shape=X.shape[:2])
    # training used pos_weight = (1 - pos) / pos, which inflates the odds by that factor; undo it so P is calibrated
    Mv = np.load(f"{a.data}/valid_M.npy", mmap_mode="r")
    pos = float(np.asarray(Mv[a.holdout:a.holdout + 200, WARMUP:]).mean()); pw = (1 - pos) / pos
    print(f"calibrating with pos_weight {pw:.3f}", flush=True)
    for s in range(0, X.shape[0], 256):
        p = predict(model, X, s, min(s + 256, X.shape[0]))
        o = p / np.clip(1 - p, 1e-6, None) / pw
        P[s:s + 256] = o / (1 + o)
        print(f"labelled {min(s + 256, X.shape[0])}/{X.shape[0]}", flush=True)
    P.flush()
    p = np.asarray(P[:, WARMUP:], dtype=np.float32)
    print(f"LABEL DONE {a.name}: mean P {p.mean():.4f}, P>0.5 {(p > 0.5).mean():.4f}", flush=True)
    raise SystemExit

X = np.load(f"{a.data}/valid_X.npy", mmap_mode="r")
M = np.load(f"{a.data}/valid_M.npy", mmap_mode="r")
tr = np.arange(a.holdout, X.shape[0])
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
nb = len(tr) // a.batch
st = dict(epoch=0, batch=0, perm=rng.permutation(tr), best=-1.0)
if a.resume and os.path.exists(f"{a.out}/latest.pt"):
    ck = torch.load(f"{a.out}/latest.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); st = ck["st"]; rng.bit_generator.state = ck["rng"]
    print(f"RESUMED epoch {st['epoch']} batch {st['batch']}/{nb} best AUC {st['best']:.4f}", flush=True)
pos = float(np.asarray(M[a.holdout:a.holdout + 200, WARMUP:]).mean())
lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=dev))
print(f"train seqs {len(tr)}, holdout {a.holdout}, scored rate {pos:.4f}", flush=True)


def save(path):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), st=st, rng=rng.bit_generator.state, cfg=vars(a)), path + ".tmp")
    os.replace(path + ".tmp", path)


t0 = time.time()
while st["epoch"] < a.epochs:
    while st["batch"] < nb:
        idx = np.sort(st["perm"][st["batch"] * a.batch:(st["batch"] + 1) * a.batch])
        x = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(dev)
        m = torch.from_numpy(np.asarray(M[idx], dtype=np.float32)).to(dev)
        h, tot = None, 0.0
        for c in range(0, L_SEQ, a.chunk):
            p, h = model(x[:, c:c + a.chunk], h); h = h.detach()
            lo = max(WARMUP - c, 0)
            loss = lossf(p[:, lo:], m[:, c + lo:c + a.chunk])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        st["batch"] += 1
        print(f"ep {st['epoch']} batch {st['batch']}/{nb} loss {tot / (L_SEQ // a.chunk):.4f} {time.time() - t0:.0f}s", flush=True)
        if st["batch"] % a.save_every == 0:
            save(f"{a.out}/latest.pt")
    p = predict(model, X, 0, a.holdout)[:, WARMUP:].ravel()
    y = np.asarray(M[:a.holdout, WARMUP:]).ravel().astype(np.int8)
    auc, ap_ = auc_ap(y, p)
    print(f"EVAL ep {st['epoch']} holdout AUC {auc:.4f} AP {ap_:.4f} (base rate {y.mean():.4f})", flush=True)
    if auc > st["best"]:
        st["best"] = auc; save(f"{a.out}/best.pt")
    st["epoch"] += 1; st["batch"] = 0; st["perm"] = rng.permutation(tr)
    save(f"{a.out}/latest.pt")
print("MASK TRAIN DONE best AUC", st["best"], flush=True)
