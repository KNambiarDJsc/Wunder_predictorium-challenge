"""
Competition inference interface.

CRITICAL GUARANTEES:
- Exact feature parity with training
- Stateful deltas and rolling regime
- O(1) per step inference
- NO hard abstention (evaluator expects dense predictions)
- Confidence gates prediction internally
- Fully aligned with utils.py evaluator
"""

import numpy as np
import onnxruntime as ort
from collections import deque


class PredictionModel:
    def __init__(self):
        # Load ONNX model
        self.session = ort.InferenceSession(
            "predictor_final.onnx",
            providers=["CPUExecutionProvider"]
        )

        self.current_seq = None
        self.reset_state()

    def reset_state(self):
        # Fixed-size rolling buffer (O(1))
        self.buffer = deque(maxlen=1000)

        # Rolling statistics (for selective normalization)
        self.feature_sum = np.zeros(64, dtype=np.float32)
        self.feature_sq_sum = np.zeros(64, dtype=np.float32)
        self.feature_mean = np.zeros(64, dtype=np.float32)
        self.feature_std = np.ones(64, dtype=np.float32)

        # Stateful history
        self.prev_mid = None
        self.prev_imbalance = None

        self.vol_sum = 0.0
        self.vol_mean = 0.0
        self.spread_sum = 0.0
        self.spread_mean = 0.0

        self.delta_mid_history = deque(maxlen=128)
        self.delta_imbalance_history = deque(maxlen=128)
        self.vol_history = deque(maxlen=128)

        self.step = 0

    def predict(self, data_point):
        """
        Args:
            data_point: object with attributes
                seq_ix, step_in_seq, need_prediction,
                p0..p11, v0..v11, dp0..dp3, dv0..dv3

        Returns:
            np.ndarray shape (2,) or None
        """

        # Reset on new sequence
        if data_point.seq_ix != self.current_seq:
            self.current_seq = data_point.seq_ix
            self.reset_state()

        # Extract raw inputs
        p = np.array([getattr(data_point, f"p{i}") for i in range(12)], dtype=np.float32)
        v = np.array([getattr(data_point, f"v{i}") for i in range(12)], dtype=np.float32)
        dp = np.array([getattr(data_point, f"dp{i}") for i in range(4)], dtype=np.float32)
        dv = np.array([getattr(data_point, f"dv{i}") for i in range(4)], dtype=np.float32)

        # Compute features (stateful, parity-safe)
        features = self._compute_features_stateful(p, v, dp, dv)

        self.buffer.append(features)
        self.step += 1

        # Warm-up: evaluator EXPECTS None here
        if not data_point.need_prediction:
            return None

        # Build model input (1, T, 64)
        x = np.array(self.buffer, dtype=np.float32)[None, :, :]

        # ONNX inference
        pred, conf = self.session.run(None, {"x": x})

        # Model already gates predictions internally using confidence
        final_pred = np.clip(pred[0], -6.0, 6.0)

        return final_pred

    def _compute_features_stateful(self, p, v, dp, dv):
        """
        Stateful feature computation.
        MUST exactly match features.py semantics.
        """
        features = np.zeros(64, dtype=np.float32)

        # =====================
        # GEOMETRY
        # =====================
        mid = (p[0] + p[6]) * 0.5
        spread = p[6] - p[0]

        weights = np.array([1.0, 0.5, 0.333, 0.25, 0.2, 0.167], dtype=np.float32)
        bid_w = np.sum(v[:6] * weights)
        ask_w = np.sum(v[6:] * weights)
        imbalance = (bid_w - ask_w) / (bid_w + ask_w + 1e-8)
        imbalance = np.clip(imbalance, -1.5, 1.5)

        features[0] = mid
        features[1] = spread
        features[2] = imbalance
        features[3] = (v[0] - v[1]) - (v[1] - v[2])
        features[4] = (v[6] - v[7]) - (v[7] - v[8])

        # =====================
        # DYNAMICS
        # =====================
        delta_mid = 0.0 if self.prev_mid is None else mid - self.prev_mid
        delta_imb = 0.0 if self.prev_imbalance is None else imbalance - self.prev_imbalance

        self.prev_mid = mid
        self.prev_imbalance = imbalance

        self.delta_mid_history.append(delta_mid)
        self.delta_imbalance_history.append(delta_imb)

        signed_pressure = 0.0
        for i in range(4):
            if dv[i] > 0:
                signed_pressure += dv[i] if dp[i] > mid else -dv[i]

        features[5] = delta_mid
        features[6] = delta_imb
        features[7] = signed_pressure

        # =====================
        # REGIME
        # =====================
        vol_proxy = (
            np.std(self.delta_mid_history)
            if len(self.delta_mid_history) >= 10
            else 0.0
        )
        self.vol_history.append(vol_proxy)

        total_vol = v[:6].sum() + v[6:].sum()
        self.vol_sum += total_vol
        self.vol_mean = self.vol_sum / max(self.step + 1, 1)

        imb_persist = (
            np.mean(np.abs(self.delta_imbalance_history))
            if len(self.delta_imbalance_history) >= 10
            else 0.0
        )

        features[8] = vol_proxy
        features[9] = total_vol / (self.vol_mean + 1e-8)
        features[10] = imb_persist

        # =====================
        # CONFIDENCE SIGNALS
        # =====================
        vol_high = 0.0
        if len(self.vol_history) >= 20:
            vh = np.array(self.vol_history)[-20:]
            vol_high = 1.0 if vol_proxy > vh.mean() + vh.std() else 0.0

        features[11] = vol_high
        features[12] = abs(delta_imb)
        features[13] = abs(signed_pressure) / (dv.sum() + 1e-8)

        # =====================
        # RAW FEATURES
        # =====================
        features[14:26] = p
        features[26:38] = v
        features[38:42] = dp
        features[42:46] = dv

        # =====================
        # RELATIVE PRICES
        # =====================
        self.spread_sum += spread
        self.spread_mean = self.spread_sum / max(self.step + 1, 1)
        features[46] = spread / (self.spread_mean + 1e-8)

        for i in range(6):
            features[47 + i] = (p[i] - mid) / (spread + 1e-8)
            features[53 + i] = (p[6 + i] - mid) / (spread + 1e-8)

        # =====================
        # SELECTIVE NORMALIZATION
        # =====================
        if self.step > 10:
            for i in (2, 5, 6):
                self.feature_sum[i] += features[i]
                self.feature_sq_sum[i] += features[i] ** 2
                mean = self.feature_sum[i] / (self.step + 1)
                var = self.feature_sq_sum[i] / (self.step + 1) - mean ** 2
                std = np.sqrt(var + 1e-8)

                self.feature_mean[i] = mean
                self.feature_std[i] = std
                features[i] = (features[i] - mean) / std

        return features
