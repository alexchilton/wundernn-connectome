"""Stateful GRU trainer for Alpha Connectome.

Training runs truncated BPTT over whole 20,000-step sequences: a batch of B sequences is walked in
chunks of C steps, carrying the (detached) hidden state, exactly as inference sees the data.
Loss is applied on steps >= 99 only. Targets are clipped to [-2, 2] as the scorer does.
  --loss mse   plain MSE
  --loss wmse  MSE weighted by |clip(y)|, the scorer's weight; the optimum is E[|y| y|x] / E[|y| | x]

Checkpoints: <out>/latest.pt every --save-every batches, <out>/best.pt on best valid WP.
--resume restores model, optimiser, scheduler, epoch, batch position, permutation and RNG.
--init loads model weights only.
Export: <out>/model.onnx + <out>/meta.json, same interface as the baseline (features, hidden_k).
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wnn_connectome_starterpack"))
from utils import weighted_pearson  # noqa: E402

WARMUP, L_SEQ, NF = 99, 20000, 112

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--train-name", default="train")
ap.add_argument("--valid-name", default="valid")
ap.add_argument("--out", required=True)
ap.add_argument("--hidden", type=int, default=128)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--loss", choices=["mse", "wmse"], default="wmse")
ap.add_argument("--chunk", type=int, default=500)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--n-train", type=int, default=0, help="use the first n train sequences (0 = all)")
ap.add_argument("--n-valid", type=int, default=0, help="use the first n valid sequences (0 = all)")
ap.add_argument("--eval-every", type=int, default=0, help="batches between evals (0 = once per epoch)")
ap.add_argument("--save-every", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
ap.add_argument("--resume", action="store_true")
ap.add_argument("--init", default="")
ap.add_argument("--max-batches", type=int, default=0, help="smoke test: stop after this many batches")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
print("CONFIG", json.dumps(vars(a)), flush=True)
dev = torch.device(a.device)


class StackedGRU(nn.Module):
    """Separate single-layer GRUs so each layer's state is its own ONNX input, like the baseline."""

    def __init__(self, hidden, layers):
        super().__init__()
        self.grus = nn.ModuleList([nn.GRU(NF if i == 0 else hidden, hidden, batch_first=True) for i in range(layers)])
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, hs):
        new = []
        for gru, h in zip(self.grus, hs):
            x, h = gru(x, h)
            new.append(h)
        return self.head(x), new


def load(name, n):
    X = np.load(f"{a.data}/{name}_X.npy", mmap_mode="r")
    Y = np.load(f"{a.data}/{name}_Y.npy", mmap_mode="r")
    M = np.load(f"{a.data}/{name}_M.npy", mmap_mode="r") if os.path.exists(f"{a.data}/{name}_M.npy") else None
    if n:
        X, Y = X[:n], Y[:n]
        M = M[:n] if M is not None else None
    return X, Y, M


Xtr, Ytr, _ = load(a.train_name, a.n_train)
Xva, Yva, Mva = load(a.valid_name, a.n_valid)
print(f"train {Xtr.shape[0]} seqs, valid {Xva.shape[0]} seqs (mask {'yes' if Mva is not None else 'no'})", flush=True)

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
model = StackedGRU(a.hidden, a.layers).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
n_batches = Xtr.shape[0] // a.batch
total = n_batches * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(total, 1), pct_start=0.05)
state = dict(epoch=0, batch=0, perm=rng.permutation(Xtr.shape[0]), best=-1.0, history=[])

if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    print("INIT weights from", a.init, flush=True)
if a.resume and os.path.exists(f"{a.out}/latest.pt"):
    ck = torch.load(f"{a.out}/latest.pt", map_location=dev, weights_only=False)
    if ck["cfg"]["hidden"] != a.hidden or ck["cfg"]["layers"] != a.layers:
        raise SystemExit("refusing to resume across a shape change")
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    state = ck["state"]; rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
    print(f"RESUMED epoch {state['epoch']} batch {state['batch']}/{n_batches}, best {state['best']:.5f}, "
          f"{len(state['history'])} evals in history, perm len {len(state['perm'])}", flush=True)


def save(path):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=state,
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cfg=vars(a)), path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def evaluate():
    model.eval()
    ys, ps = [], []
    for s in range(0, Xva.shape[0], 64):
        x = torch.from_numpy(np.asarray(Xva[s:s + 64], dtype=np.float32)).to(dev)
        hs = [torch.zeros(1, x.shape[0], a.hidden, device=dev) for _ in range(a.layers)]
        out = []
        for c in range(0, L_SEQ, 2000):
            p, hs = model(x[:, c:c + 2000], hs)
            out.append(p.float().cpu().numpy())
        p = np.concatenate(out, 1)
        y = np.asarray(Yva[s:s + 64])
        m = np.zeros(y.shape[:2], bool); m[:, WARMUP:] = True
        if Mva is not None:
            m &= np.asarray(Mva[s:s + 64])
        ys.append(y[m]); ps.append(p[m])
    y, p = np.concatenate(ys), np.concatenate(ps)
    model.train()
    wp = [weighted_pearson(y[:, k], p[:, k]) for k in range(2)]
    return float(np.mean(wp)), wp


def loss_fn(p, y, first_step):
    y = y.clamp(-2, 2)
    w = y.abs() if a.loss == "wmse" else torch.ones_like(y)
    if first_step < WARMUP:
        keep = torch.arange(first_step, first_step + y.shape[1], device=y.device) >= WARMUP
        w = w * keep[None, :, None]
    return (w * (p - y) ** 2).sum() / w.sum().clamp_min(1e-6)


t0 = time.time()
done_batches = state["epoch"] * n_batches + state["batch"]
while state["epoch"] < a.epochs:
    while state["batch"] < n_batches:
        idx = np.sort(state["perm"][state["batch"] * a.batch:(state["batch"] + 1) * a.batch])
        x = torch.from_numpy(np.asarray(Xtr[idx], dtype=np.float32)).to(dev)
        y = torch.from_numpy(np.asarray(Ytr[idx], dtype=np.float32)).to(dev)
        hs = [torch.zeros(1, len(idx), a.hidden, device=dev) for _ in range(a.layers)]
        tot = 0.0
        for c in range(0, L_SEQ, a.chunk):
            p, hs = model(x[:, c:c + a.chunk], hs)
            hs = [h.detach() for h in hs]
            loss = loss_fn(p, y[:, c:c + a.chunk], c)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        sched.step()
        state["batch"] += 1; done_batches += 1
        el = time.time() - t0
        print(f"ep {state['epoch']} batch {state['batch']}/{n_batches} loss {tot / (L_SEQ // a.chunk):.5f} "
              f"lr {sched.get_last_lr()[0]:.2e} {el:.0f}s", flush=True)
        if a.eval_every and state["batch"] % a.eval_every == 0:
            wp, per = evaluate(); state["history"].append((state["epoch"], state["batch"], wp))
            print(f"EVAL ep {state['epoch']} batch {state['batch']} WP {wp:.5f} (t0 {per[0]:.5f} t1 {per[1]:.5f})", flush=True)
            if wp > state["best"]:
                state["best"] = wp; save(f"{a.out}/best.pt")
        if state["batch"] % a.save_every == 0:
            save(f"{a.out}/latest.pt")
        if a.max_batches and done_batches >= a.max_batches:
            break
    if a.max_batches and done_batches >= a.max_batches:
        save(f"{a.out}/latest.pt")
        break
    wp, per = evaluate(); state["history"].append((state["epoch"], state["batch"], wp))
    print(f"EVAL ep {state['epoch']} end WP {wp:.5f} (t0 {per[0]:.5f} t1 {per[1]:.5f})", flush=True)
    if wp > state["best"]:
        state["best"] = wp; save(f"{a.out}/best.pt")
    state["epoch"] += 1; state["batch"] = 0; state["perm"] = rng.permutation(Xtr.shape[0])
    save(f"{a.out}/latest.pt")

# export best (or latest for a smoke run) to ONNX with the baseline's interface
src = f"{a.out}/best.pt" if os.path.exists(f"{a.out}/best.pt") else f"{a.out}/latest.pt"
model.load_state_dict(torch.load(src, map_location="cpu", weights_only=False)["model"])
model = model.cpu().eval()


class Step(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, features, *hidden):
        p, hs = self.m(features, list(hidden))
        return (p, *hs)


names = [f"hidden_{i}" for i in range(a.layers)]
dummy = (torch.zeros(1, 1, NF), *[torch.zeros(1, 1, a.hidden) for _ in range(a.layers)])
torch.onnx.export(Step(model), dummy, f"{a.out}/model.onnx", input_names=["features", *names],
                  output_names=["prediction", *[f"next_{n}" for n in names]], opset_version=17, dynamo=False)
json.dump(dict(hidden=a.hidden, layers=a.layers, best_wp=state["best"], source=src, cfg=vars(a)),
          open(f"{a.out}/meta.json", "w"), indent=1)
print(f"EXPORTED {a.out}/model.onnx from {src}, best WP {state['best']:.5f}", flush=True)
print("TRAIN DONE", flush=True)
