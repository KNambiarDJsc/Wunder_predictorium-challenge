import os
import sys
import numpy as np
import onnxruntime as ort
from collections import deque

# Allow importing utils during local testing (ignored by platform)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

try:
    from utils import DataPoint, ScorerStepByStep
except ImportError:
    DataPoint = None
    ScorerStepByStep = None


class PredictionModel:
    """
    Competition inference model.

    CRITICAL CONTRACT:
    - Uses data_point.state ONLY
    - No feature recomputation
    - Returns None until need_prediction=True
    - Returns np.ndarray shape (2,)
    """

    def __init__(self):
        self.current_seq_ix = None
        self.buffer = []

        # Resolve ONNX path relative to solution.py
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "predictor_final.onnx")

        # ONNX Runtime session (CPU only, deterministic)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self.session = ort.InferenceSession(
                onnx_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, data_point):
        """
        Args:
            data_point:
                - seq_ix
                - step_in_seq
                - need_prediction
                - state: np.ndarray (features)

        Returns:
            np.ndarray (2,) or None
        """

        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.buffer = []

        # Append feature state (DO NOT MODIFY)
        self.buffer.append(data_point.state.astype(np.float32, copy=True))

        # Warm-up phase: evaluator expects None
        if not data_point.need_prediction:
            return None

        # Build model input: (1, T, F)
        x = np.asarray(self.buffer, dtype=np.float32)[None, :, :]

        # ONNX inference
        pred, conf = self.session.run(
            self.output_names,
            {self.input_name: x}
        )

        # Model already applies confidence gating internally
        return np.clip(pred[0], -6.0, 6.0)


# ==========================
# Optional local validation
# ==========================
if __name__ == "__main__":
    if ScorerStepByStep is None:
        print("utils.py not available — skipping local test")
    else:
        valid_path = os.path.join(CURRENT_DIR, "..", "data", "valid.parquet")
        if os.path.exists(valid_path):
            model = PredictionModel()
            scorer = ScorerStepByStep(valid_path)

            print("Running local scorer-style validation...")
            results = scorer.score(model)

            print("\nResults:")
            print(f"Weighted Pearson: {results['weighted_pearson']:.6f}")
            for k, v in results.items():
                if k != "weighted_pearson":
                    print(f"{k}: {v:.6f}")
        else:
            print("valid.parquet not found — skipping test")
