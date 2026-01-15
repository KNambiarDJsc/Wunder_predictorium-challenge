import os
import sys
import numpy as np
import onnxruntime as ort

# Allow importing utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint


class PredictionModel:
    """
    Competition-compatible inference wrapper.
    STRICTLY follows the official baseline interface.
    """

    def __init__(self):
        self.current_seq_ix = None
        self.sequence_history = []

        # Load ONNX from submission directory
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

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict(self, data_point: DataPoint):
        """
        Args:
            data_point.state : np.ndarray (feature vector)
            data_point.need_prediction : bool
        """

        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []

        # Append current feature vector
        self.sequence_history.append(data_point.state.copy())

        # Warm-up period: MUST return None
        if not data_point.need_prediction:
            return None

        # Build window (last 1000 steps or whatever model expects)
        window = self.sequence_history[-1000:]

        # Pad if needed (safety)
        if len(window) < 1000:
            pad = [np.zeros_like(window[0])] * (1000 - len(window))
            window = pad + window

        x = np.asarray(window, dtype=np.float32)[None, :, :]

        # ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: x})

        # Predictor outputs (pred, conf)
        pred = outputs[0][0]

        # Clamp per competition rules
        return np.clip(pred, -6.0, 6.0)
