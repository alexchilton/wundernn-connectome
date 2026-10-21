"""Fuse two plain GRU checkpoints (same hidden, layers, state_reset, no extra inputs) into ONE wider GRU whose output
is exactly the mean of the two members (block-diagonal recurrent weights, shared input, averaged head).
One 256-wide GRU op costs 21.7 us/call on the M4 vs 25.9 for two ops (docs/fable-ideas.md round 2, idea 1).

    python src/fuse.py out/final_s1/best.pt out/final_s2/best.pt out/fused_s1s2
"""
import os, shutil, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
from model import Net, model_cfg, export  # noqa: E402

a_path, b_path, out = sys.argv[1:4]
ca, cb = (torch.load(p, map_location="cpu", weights_only=False) for p in (a_path, b_path))
ma, mb = model_cfg(ca["cfg"]), model_cfg(cb["cfg"])
assert ma == mb, f"members differ: {ma} vs {mb}"
assert not (ma["diff"] or ma["ema"] or ma["proj"] or ma["skip"] or ma["time"]), "plain GRU members only"
H, L = ma["hidden"], ma["layers"]
mf = dict(ma, hidden=2 * H)
fused = Net(mf)
sa, sb, sf = ca["model"], cb["model"], {}
for i in range(L):
    p = f"grus.{i}."
    wih_a, wih_b = sa[p + "weight_ih_l0"], sb[p + "weight_ih_l0"]
    wih = torch.zeros(3 * 2 * H, wih_a.shape[1] if i == 0 else 2 * H)
    whh = torch.zeros(3 * 2 * H, 2 * H)
    bih, bhh = torch.zeros(3 * 2 * H), torch.zeros(3 * 2 * H)
    for g in range(3):   # gate blocks r, z, n; member A takes units 0:H, member B units H:2H
        ra, rb, sl = slice(g * 2 * H, g * 2 * H + H), slice(g * 2 * H + H, (g + 1) * 2 * H), slice(g * H, (g + 1) * H)
        if i == 0:       # shared raw input
            wih[ra], wih[rb] = wih_a[sl], wih_b[sl]
        else:            # input is the previous fused layer: block-diagonal
            wih[ra, :H], wih[rb, H:] = wih_a[sl], wih_b[sl]
        whh[ra, :H], whh[rb, H:] = sa[p + "weight_hh_l0"][sl], sb[p + "weight_hh_l0"][sl]
        bih[ra], bih[rb] = sa[p + "bias_ih_l0"][sl], sb[p + "bias_ih_l0"][sl]
        bhh[ra], bhh[rb] = sa[p + "bias_hh_l0"][sl], sb[p + "bias_hh_l0"][sl]
    sf.update({p + "weight_ih_l0": wih, p + "weight_hh_l0": whh, p + "bias_ih_l0": bih, p + "bias_hh_l0": bhh})
sf["head.weight"] = torch.cat([sa["head.weight"], sb["head.weight"]], 1) / 2
sf["head.bias"] = (sa["head.bias"] + sb["head.bias"]) / 2
fused.load_state_dict(sf)

# parity: fused output == mean of the members on random rows, run as one sequence
na, nb = Net(ma), Net(mb)
na.load_state_dict(sa); nb.load_state_dict(sb)
x = torch.randn(1, 300, 112)
z = lambda n, h: [torch.zeros(1, 1, h) for _ in range(L)]
with torch.no_grad():
    err = (fused(x, z(fused, 2 * H))[0] - (na(x, z(na, H))[0] + nb(x, z(nb, H))[0]) / 2).abs().max().item()
print(f"parity max abs err {err:.2e}")
assert err < 1e-5
os.makedirs(out, exist_ok=True)
export(fused, mf, out, dict(members=[a_path, b_path], member_best=[ca["state"]["best"], cb["state"]["best"]]))
shutil.copy(os.path.join(os.path.dirname(__file__), "solution.py"), out)
print("fused", a_path, "+", b_path, "->", out)
