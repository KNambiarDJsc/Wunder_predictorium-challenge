import os
import sys
import numpy as np
import onnxruntime as ort

# Allow importing utils
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, ".."))

from utils import DataPoint


class PredictionModel:
    """
    Competition submission model.

    CRITICAL:
    - Uses ONLY data_point.state
    - Stateful across sequence
    - No feature recomputation
    - ONNX-safe
    """

    def __init__(self):
        self.current_seq = None
        self.buffer = []

        # Load ONNX model from SAME directory as solution.py
        onnx_path = os.path.join(CURRENT_DIR, "predictor_final.onnx")

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def reset(self):
        self.buffer = []

    def predict(self, data_point: DataPoint):
        # Reset state on new sequence
        if self.current_seq != data_point.seq_ix:
            self.current_seq = data_point.seq_ix
            self.reset()

        # Append feature vector (state is already processed)
        self.buffer.append(data_point.state.astype(np.float32))

        # Evaluator expects None before prediction window
        if not data_point.need_prediction:
            return None

        # Build input tensor (1, T, F)
        x = np.expand_dims(np.stack(self.buffer, axis=0), axis=0)

        # Run ONNX inference
        pred, conf = self.session.run(self.output_names, {self.input_name: x})

        # Return clipped prediction
        return np.clip(pred[0], -6.0, 6.0)
