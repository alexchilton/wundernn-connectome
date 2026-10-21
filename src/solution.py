"""Stateful ONNX submission: loads model.onnx and meta.json from the same directory.

meta.json lists every state tensor the graph carries: "zero" states start at 0, "x" states start from the
first row of the sequence (tiled), and states with reset=True are zeroed every state_reset rows.
"""
from pathlib import Path
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


class PredictionModel:
    def __init__(self):
        meta = json.loads((HERE / "meta.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(HERE / "model.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.spec = meta.get("states") or [dict(name=f"hidden_{i}", shape=[1, 1, meta["hidden"]], init="zero", reset=True)
                                           for i in range(meta["layers"])]
        self.names = [s["name"] for s in self.spec]
        self.state_reset = int(meta.get("state_reset", 0))
        self.seq_ix = None
        # IOBinding with two preallocated state buffers used ping-pong: call k reads bufs[k] and writes bufs[1-k].
        # Nothing is allocated or copied per call (M4: full1 14.5 -> 10.8 us/call, fused 256-wide 24.5 -> 19.7, outputs identical).
        self.x = np.zeros((1, 1, 112), dtype=np.float32)
        self.pred = np.zeros((1, 1, 2), dtype=np.float32)
        self.bufs = [[np.zeros(s["shape"], dtype=np.float32) for s in self.spec] for _ in range(2)]
        self.binds = []
        for k in range(2):
            b = self.session.io_binding()
            b.bind_cpu_input("features", self.x)
            for n, v in zip(self.names, self.bufs[k]):
                b.bind_cpu_input(n, v)
            b.bind_output("prediction", "cpu", 0, np.float32, list(self.pred.shape), self.pred.ctypes.data)
            for n, v in zip(self.names, self.bufs[1 - k]):
                b.bind_output(f"next_{n}", "cpu", 0, np.float32, list(v.shape), v.ctypes.data)
            self.binds.append(b)
        self.k = 0

    def predict(self, data_point):
        self.x[...] = np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, 112)
        cur = self.bufs[self.k]
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for s, v in zip(self.spec, cur):
                v[...] = np.tile(self.x, (1, 1, s.get("tile", 1))) if s["init"] == "x" else 0.0
        elif self.state_reset and data_point.step_in_seq % self.state_reset == 0:
            for s, v in zip(self.spec, cur):
                if s["reset"]:
                    v[...] = 0.0
        self.session.run_with_iobinding(self.binds[self.k])
        self.k ^= 1
        if not data_point.need_prediction:
            return None
        return self.pred[0, 0].copy()

if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    print(ScorerStepByStep(ap.parse_args().validation).score(PredictionModel()))
