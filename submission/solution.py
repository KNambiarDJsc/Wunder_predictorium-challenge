import os
import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        # Resolve ONNX path relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        onnx_path = os.path.join(base_dir, "predictor_final.onnx")

        # ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"]
        )

        # Cache input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Stateful buffer
        self.current_seq_ix = None
        self.history = []

    def predict(self, data_point):
        # Reset on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []

        # Append state (this is the ONLY allowed feature source)
        self.history.append(data_point.state.astype(np.float32))

        # No prediction during warmup
        if not data_point.need_prediction:
            return None

        # Build input tensor: (1, T, F)
        x = np.expand_dims(np.asarray(self.history, dtype=np.float32), axis=0)

        # Run ONNX
        outputs = self.session.run(None, {self.input_name: x})

        # Handle models with (pred, conf) or just pred
        if len(outputs) == 2:
            pred = outputs[0]   # (1, 2)
        else:
            pred = outputs[0]

        # Return (2,)
        return np.clip(pred[0], -6.0, 6.0)
