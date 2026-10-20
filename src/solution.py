"""Stateful ONNX GRU submission: loads model.onnx and meta.json from the same directory."""
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
        self.names = [f"hidden_{i}" for i in range(meta["layers"])]
        self.hidden = [np.zeros((1, 1, meta["hidden"]), dtype=np.float32) for _ in self.names]
        self.seq_ix = None

    def predict(self, data_point):
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            for h in self.hidden:
                h.fill(0.0)
        feeds = {"features": np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, 112)}
        feeds.update(zip(self.names, self.hidden))
        out = self.session.run(None, feeds)
        self.hidden = out[1:]
        if not data_point.need_prediction:
            return None
        return np.asarray(out[0][0, 0], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, str(HERE.parents[1] / "wnn_connectome_starterpack"))
    from utils import ScorerStepByStep
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", required=True)
    print(ScorerStepByStep(ap.parse_args().validation).score(PredictionModel()))
