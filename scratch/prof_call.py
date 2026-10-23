import sys, time, types, shutil, numpy as np, importlib.util, json
d = sys.argv[1]
spec = importlib.util.spec_from_file_location('sol', f'{d}/solution.py'); sol = importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
m = sol.PredictionModel()
X = np.load('out/data/valid_X.npy', mmap_mode='r')[0, :20000].astype(np.float32)
N = 20000
dp = types.SimpleNamespace(seq_ix=0, step_in_seq=0, need_prediction=True, state=None)
t = time.perf_counter()
for i in range(N):
    dp.step_in_seq = i; dp.state = X[i]; m.predict(dp)
full = (time.perf_counter() - t) / N * 1e6
b = m.binds[0]
t = time.perf_counter()
for i in range(N): m.session.run_with_iobinding(m.binds[i & 1])
ort_only = (time.perf_counter() - t) / N * 1e6
import onnx
g = onnx.load(f'{d}/model.onnx').graph
ops = {}
for n in g.node: ops[n.op_type] = ops.get(n.op_type, 0) + 1
print(f'{d}: full predict {full:.1f} us/call | ORT run only {ort_only:.1f} | python wrapper {full - ort_only:.1f} | nodes {len(g.node)} {dict(sorted(ops.items(), key=lambda x: -x[1])[:10])}')
