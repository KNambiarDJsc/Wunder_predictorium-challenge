"""
Supervised predictor model.

CRITICAL DESIGN:
- Predict ONLY last timestep
- Confidence gates prediction internally
- Regime modulates latent state via FiLM
- Architecture aligned with weighted Pearson
"""

import torch
import torch.nn as nn



class Predictor(nn.Module):
    def __init__(
        self,
        encoder,
        d_input=64,
        d_regime=12,
        freeze_encoder=True,
        unfreeze_gru=False,
    ):
        super().__init__()

        self.encoder = encoder
        self.d_gru = encoder.gru.hidden_size  # <-- FIXED (no hardcoding)


        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

            if unfreeze_gru:
                for p in self.encoder.gru.parameters():
                    p.requires_grad = True


        self.regime_proj = nn.Sequential(
            nn.Linear(self.d_gru + d_input, d_regime),
            nn.GELU()
        )


        self.film_scale = nn.Linear(d_regime, self.d_gru)
        self.film_shift = nn.Linear(d_regime, self.d_gru)


        self.pred_head_t0 = nn.Sequential(
            nn.Linear(self.d_gru, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )

        self.pred_head_t1 = nn.Sequential(
            nn.Linear(self.d_gru, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )


        self.conf_head = nn.Sequential(
            nn.Linear(self.d_gru + d_regime, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, F)

        Returns:
            pred: (B, 2)  - gated predictions [t0, t1]
            conf: (B, 1)  - confidence
        """

        z = self.encoder(x)                  # (B, T, d_gru)

        z_last = z[:, -1, :]                 # (B, d_gru)
        x_last = x[:, -1, :]                 # (B, F)


        regime_input = torch.cat([z_last, x_last], dim=1)
        regime = self.regime_proj(regime_input)  # (B, d_regime)


        scale = torch.tanh(self.film_scale(regime))
        shift = self.film_shift(regime)

        z_mod = z_last * (1.0 + scale) + shift  # (B, d_gru)


        t0 = self.pred_head_t0(z_mod)
        t1 = self.pred_head_t1(z_mod)
        pred = torch.cat([t0, t1], dim=1)       # (B, 2)


        conf_input = torch.cat([z_mod, regime], dim=1)
        conf = self.conf_head(conf_input)       # (B, 1)

        pred = pred * conf                      # gate internally
        pred = torch.clamp(pred, -6.0, 6.0)     # safety clip

        return pred, conf




class PredictorForONNX(nn.Module):
    """
    ONNX-safe wrapper.
    Always returns last-timestep prediction.
    """
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, x):
        return self.predictor(x)
