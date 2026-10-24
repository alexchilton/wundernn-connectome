"""Per-step latency, 1 ORT thread: fused ONNX GRU op vs unrolled MatMul GRU (fp32 and dynamic INT8). Random weights."""
import time, numpy as np, torch, torch.nn as nn, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
NF, OUT = 112, 2
class FusedStep(nn.Module):
    def __init__(s, H, L): super().__init__(); s.g = nn.ModuleList([nn.GRU(NF if i == 0 else H, H, batch_first=True) for i in range(L)]); s.head = nn.Linear(H, OUT)
    def forward(s, x, *hs):
        nh = []
        for g, h in zip(s.g, hs): x, h2 = g(x, h); nh.append(h2)
        return (s.head(x), *nh)
class UnrolledStep(nn.Module):   # same GRU equations (PyTorch gate order r, z, n) as explicit MatMuls
    def __init__(s, H, L):
        super().__init__(); s.H = H
        s.wi = nn.ModuleList([nn.Linear(NF if i == 0 else H, 3 * H) for i in range(L)]); s.wh = nn.ModuleList([nn.Linear(H, 3 * H) for _ in range(L)]); s.head = nn.Linear(H, OUT)
    def forward(s, x, *hs):
        x = x[:, 0]; nh = []; H = s.H
        for wi, wh, h in zip(s.wi, s.wh, hs):
            h = h[0]; gi, gh = wi(x), wh(h)
            r = torch.sigmoid(gi[:, :H] + gh[:, :H]); z = torch.sigmoid(gi[:, H:2 * H] + gh[:, H:2 * H])
            n = torch.tanh(gi[:, 2 * H:] + r * gh[:, 2 * H:]); h = (1 - z) * n + z * h
            x = h; nh.append(h[None])
        return (s.head(x)[:, None], *nh)
def bench(path, H, L, n=20000):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    feeds = {'x': np.random.randn(1, 1, NF).astype(np.float32), **{f'h{i}': np.zeros((1, 1, H), np.float32) for i in range(L)}}
    for _ in range(2000): s.run(None, feeds)
    best = 1e9
    for _ in range(3):
        t = time.perf_counter()
        for _ in range(n): s.run(None, feeds)
        best = min(best, (time.perf_counter() - t) / n * 1e6)
    return best
def export(m, H, L, path):
    dummy = (torch.zeros(1, 1, NF), *[torch.zeros(1, 1, H) for _ in range(L)])
    torch.onnx.export(m.eval(), dummy, path, input_names=['x'] + [f'h{i}' for i in range(L)], output_names=['p'] + [f'nh{i}' for i in range(L)], opset_version=17, dynamo=False)
torch.manual_seed(0)
for H in (256, 384, 512):
    L = 2; f, u = FusedStep(H, L), UnrolledStep(H, L)
    # parity: copy fused weights into the unrolled module
    for i in range(L):
        u.wi[i].weight.data = f.g[i].weight_ih_l0.data.clone(); u.wi[i].bias.data = f.g[i].bias_ih_l0.data.clone()
        u.wh[i].weight.data = f.g[i].weight_hh_l0.data.clone(); u.wh[i].bias.data = f.g[i].bias_hh_l0.data.clone()
    u.head.load_state_dict(f.head.state_dict())
    x = torch.randn(1, 1, NF); hs = [torch.randn(1, 1, H) for _ in range(L)]
    with torch.no_grad(): d = max((a - b).abs().max().item() for a, b in zip(f(x, *hs), u(x, *hs)))
    pf, pu, pq = f'/tmp/b_f{H}.onnx', f'/tmp/b_u{H}.onnx', f'/tmp/b_q{H}.onnx'
    export(f, H, L, pf); export(u, H, L, pu)
    quantize_dynamic(pu, pq, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul', 'Gemm'])
    sess = lambda p: ort.InferenceSession(p, providers=['CPUExecutionProvider'])
    feeds = {'x': x.numpy(), **{f'h{i}': hs[i].numpy() for i in range(L)}}
    qd = np.abs(sess(pq).run(None, feeds)[0] - sess(pf).run(None, feeds)[0]).max()
    import os
    print(f'H={H} L={L}: parity fused-vs-unrolled {d:.2e}, int8 output diff {qd:.3e} | us/step fused {bench(pf, H, L):.1f}  unrolled fp32 {bench(pu, H, L):.1f}  unrolled int8 {bench(pq, H, L):.1f} | MB {os.path.getsize(pf)/1e6:.2f} / {os.path.getsize(pq)/1e6:.2f}', flush=True)
