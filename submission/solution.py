import os
import sys
import numpy as np
import onnxruntime as ort

# Allow importing utils when testing locally (ignored by platform)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint  # used only for local testing


class PredictionModel:
    """
    Competition inference wrapper.
    Uses precomputed features from data_point.state.
    """

    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []

        # Load ONNX model from same directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "predictor_final.onnx")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Resolve input/output names dynamically (VERY IMPORTANT)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, data_point):
        """
        Args:
            data_point.state: np.ndarray (feature vector)
            data_point.need_prediction: bool

        Returns:
            None OR np.ndarray shape (2,)
        """

        # Reset sequence buffer
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Append feature vector
        self.sequence_history.append(data_point.state.astype(np.float32))

        # Warm-up phase
        if not data_point.need_prediction:
            return None

        # Build model input (1, T, F)
        x = np.asarray(self.sequence_history, dtype=np.float32)[None, :, :]

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: x})

        # Predictor returns (pred, conf) OR (pred)
        pred = outputs[0]

        # Shape handling
        if pred.ndim == 3:
            pred = pred[0, -1]
        else:
            pred = pred[0]

        # Safety clip
        pred = np.clip(pred, -6.0, 6.0)

        return pred.astype(np.float32)
