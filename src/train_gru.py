"""Stateful GRU trainer for Alpha Connectome.

Training runs truncated BPTT over whole 20,000-step sequences: a batch of B sequences is walked in
chunks of C steps, carrying the (detached) hidden state, exactly as inference sees the data.
Loss is applied on steps >= 99 only. Targets are clipped to [-2, 2] as the scorer does.
  --loss mse   plain MSE
  --loss wmse  MSE weighted by |clip(y)|, the scorer's weight; the optimum is E[|y| y|x] / E[|y| | x]

Checkpoints: <out>/latest.pt every --save-every batches, <out>/best.pt on best valid WP.
--resume restores model, optimiser, scheduler, epoch, batch position, permutation and RNG.
--init loads model weights only.
Export: <out>/model.onnx + <out>/meta.json via model.export (features in, explicit state tensors in/out).
Architecture options (--diff, --ema, --proj, --skip) are documented in model.py.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "wnn_connectome_starterpack"))
from utils import weighted_pearson  # noqa: E402
from model import Net, model_cfg, make_feats, export  # noqa: E402

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
ap.add_argument("--soft-mask", action="store_true", help="weight the train loss by <data>/<train-name>_P.npy (mask model P(scored))")
ap.add_argument("--add-valid-from", type=int, default=0, help="also train on valid[k:] with its true mask (0 = off)")
ap.add_argument("--state-reset", type=int, default=0, help="zero the GRU state every N rows, in training, eval and export (0 = never)")
ap.add_argument("--diff", action="store_true", help="add x - x_prev to the inputs")
ap.add_argument("--ema", default="", help="comma list of EMA time constants; adds x - ema_T(x) per T")
ap.add_argument("--proj", type=int, default=0, help="Linear input projection width (0 = none)")
ap.add_argument("--skip", action="store_true", help="head also sees the raw row")
ap.add_argument("--time", action="store_true", help="add log(1 + step) / log(20000) as an input")
ap.add_argument("--loss-norm", choices=["chunk", "batch"], default="chunk",
                help="chunk: divide each chunk's loss by its own weight sum; batch: by the batch's mean weight per chunk, "
                     "so chunks count in proportion to their weight mass, as rows do in the global metric")
ap.add_argument("--ema-weights", type=float, default=0.0, help="keep an EMA of the weights with this decay per optimiser step; "
                "evaluate, save as best and export the EMA weights (0 = off)")
ap.add_argument("--max-batches", type=int, default=0, help="smoke test: stop after this many batches")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
print("CONFIG", json.dumps(vars(a)), flush=True)
dev = torch.device(a.device)


def load(name, n):
    X = np.load(f"{a.data}/{name}_X.npy", mmap_mode="r")
    Y = np.load(f"{a.data}/{name}_Y.npy", mmap_mode="r")
    M = np.load(f"{a.data}/{name}_M.npy", mmap_mode="r") if os.path.exists(f"{a.data}/{name}_M.npy") else None
    if n:
        X, Y = X[:n], Y[:n]
        M = M[:n] if M is not None else None
    return X, Y, M


Xtr, Ytr, _ = load(a.train_name, a.n_train)
Ptr = None
if a.soft_mask:
    Ptr = np.load(f"{a.data}/{a.train_name}_P.npy", mmap_mode="r")[:Xtr.shape[0]]
Xva, Yva, Mva = load(a.valid_name, a.n_valid)
# training pool: (X, Y, W) sources; W is a per-row weight multiplier (None = 1)
sources = [(Xtr, Ytr, Ptr)]
if a.add_valid_from:
    Xv, Yv, Mv = load(a.valid_name, 0)
    sources.append((Xv[a.add_valid_from:], Yv[a.add_valid_from:], Mv[a.add_valid_from:]))
pool = [(si, i) for si, (X, _, _) in enumerate(sources) for i in range(X.shape[0])]
if a.state_reset:
    assert a.state_reset % a.chunk == 0 and 2000 % a.state_reset == 0 or a.state_reset % 2000 == 0, "state-reset must align with chunk and eval chunks"
print(f"train pool {len(pool)} seqs ({' + '.join(str(X.shape[0]) for X, _, _ in sources)}), soft mask {a.soft_mask}, "
      f"state reset {a.state_reset}, valid {Xva.shape[0]} seqs (mask {'yes' if Mva is not None else 'no'})", flush=True)

torch.manual_seed(a.seed)
rng = np.random.default_rng(a.seed)
MC = model_cfg(vars(a))
print("MODEL", json.dumps(MC), flush=True)
model = Net(MC).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
ema_model = None
if a.ema_weights:
    from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(a.ema_weights))
n_batches = len(pool) // a.batch
total = n_batches * a.epochs
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(total, 1), pct_start=0.05)
state = dict(epoch=0, batch=0, perm=rng.permutation(len(pool)), best=-1.0, history=[])

if a.init:
    model.load_state_dict(torch.load(a.init, map_location=dev)["model"])
    print("INIT weights from", a.init, flush=True)
if a.resume and os.path.exists(f"{a.out}/latest.pt"):
    ck = torch.load(f"{a.out}/latest.pt", map_location=dev, weights_only=False)
    if model_cfg(ck["cfg"]) != MC:
        raise SystemExit("refusing to resume across a shape change")
    model.load_state_dict(ck["raw_model"] if "raw_model" in ck else ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    if ema_model is not None:
        if "ema_model" not in ck:
            raise SystemExit("refusing to resume an EMA run from a checkpoint without EMA state")
        ema_model.load_state_dict(ck["ema_model"])
    state = ck["state"]; rng.bit_generator.state = ck["rng"]; torch.set_rng_state(ck["torch_rng"])
    print(f"RESUMED epoch {state['epoch']} batch {state['batch']}/{n_batches}, best {state['best']:.5f}, "
          f"{len(state['history'])} evals in history, perm len {len(state['perm'])}", flush=True)


def save(path):
    # "model" is what gets evaluated and exported: the EMA weights when --ema-weights is on
    extra = dict(raw_model=model.state_dict(), ema_model=ema_model.state_dict()) if ema_model is not None else {}
    torch.save(dict(model=(ema_model.module if ema_model is not None else model).state_dict(), **extra,
                    opt=opt.state_dict(), sched=sched.state_dict(), state=state,
                    rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(), cfg=vars(a)), path + ".tmp")
    os.replace(path + ".tmp", path)


@torch.no_grad()
def evaluate():
    net = ema_model.module if ema_model is not None else model
    net.eval()
    ys, ps = [], []
    for s in range(0, Xva.shape[0], 64):
        x = torch.from_numpy(make_feats(np.asarray(Xva[s:s + 64], dtype=np.float32), MC)).to(dev)
        hs = [torch.zeros(1, x.shape[0], a.hidden, device=dev) for _ in range(a.layers)]
        out = []
        step = min(2000, a.state_reset) if a.state_reset else 2000
        for c in range(0, L_SEQ, step):
            if a.state_reset and c % a.state_reset == 0:
                hs = [torch.zeros_like(h) for h in hs]
            p, hs = net(x[:, c:c + step], hs)
            out.append(p.float().cpu().numpy())
        p = np.concatenate(out, 1)
        y = np.asarray(Yva[s:s + 64])
        m = np.zeros(y.shape[:2], bool); m[:, WARMUP:] = True
        if Mva is not None:
            m &= np.asarray(Mva[s:s + 64])
        ys.append(y[m]); ps.append(p[m])
    y, p = np.concatenate(ys), np.concatenate(ps)
    net.train(); model.train()
    wp = [weighted_pearson(y[:, k], p[:, k]) for k in range(2)]
    return float(np.mean(wp)), wp


def loss_fn(p, y, first_step, rw, denom=None):
    y = y.clamp(-2, 2)
    w = y.abs() if a.loss == "wmse" else torch.ones_like(y)
    w = w * rw[:, :, None]
    if first_step < WARMUP:
        keep = torch.arange(first_step, first_step + y.shape[1], device=y.device) >= WARMUP
        w = w * keep[None, :, None]
    return (w * (p - y) ** 2).sum() / (denom if denom is not None else w.sum().clamp_min(1e-6))


t0 = time.time()
done_batches = state["epoch"] * n_batches + state["batch"]
while state["epoch"] < a.epochs:
    while state["batch"] < n_batches:
        idx = sorted(pool[j] for j in state["perm"][state["batch"] * a.batch:(state["batch"] + 1) * a.batch])
        x = torch.from_numpy(make_feats(np.stack([np.asarray(sources[si][0][i], dtype=np.float32) for si, i in idx]), MC)).to(dev)
        y = torch.from_numpy(np.stack([np.asarray(sources[si][1][i], dtype=np.float32) for si, i in idx])).to(dev)
        rw = torch.from_numpy(np.stack([np.ones(L_SEQ, np.float32) if sources[si][2] is None
                                        else np.asarray(sources[si][2][i], dtype=np.float32) for si, i in idx])).to(dev)
        hs = [torch.zeros(1, len(idx), a.hidden, device=dev) for _ in range(a.layers)]
        tot = 0.0
        denom = None
        if a.loss_norm == "batch":   # the batch's total loss weight (after warm-up and mask) spread evenly over its chunks
            wy = y.clamp(-2, 2).abs() if a.loss == "wmse" else torch.ones_like(y)
            wy = wy * rw[:, :, None]; wy[:, :WARMUP] = 0
            denom = (wy.sum() / (L_SEQ // a.chunk)).clamp_min(1e-6)
        for c in range(0, L_SEQ, a.chunk):
            if a.state_reset and c % a.state_reset == 0:
                hs = [torch.zeros_like(h) for h in hs]
            p, hs = model(x[:, c:c + a.chunk], hs)
            hs = [h.detach() for h in hs]
            loss = loss_fn(p, y[:, c:c + a.chunk], c, rw[:, c:c + a.chunk], denom)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
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
    state["epoch"] += 1; state["batch"] = 0; state["perm"] = rng.permutation(len(pool))
    save(f"{a.out}/latest.pt")

# export best (or latest for a smoke run) to ONNX
src = f"{a.out}/best.pt" if os.path.exists(f"{a.out}/best.pt") else f"{a.out}/latest.pt"
model.load_state_dict(torch.load(src, map_location="cpu", weights_only=False)["model"])   # EMA weights if --ema-weights
export(model, MC, a.out, dict(best_wp=state["best"], source=src, cfg=vars(a)))
print(f"EXPORTED {a.out}/model.onnx from {src}, best WP {state['best']:.5f}", flush=True)
print("TRAIN DONE", flush=True)
