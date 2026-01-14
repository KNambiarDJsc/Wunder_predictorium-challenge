"""
Self-supervised temporal encoder with structurally correct masking.

KEY PROPERTIES:
- Contiguous temporal masking (biased toward recent timesteps)
- Feature-group masking (structured, not random noise)
- Strong reconstruction head
- Loss computed ONLY on masked positions
- Feature-wise scale normalization
"""

import torch
import torch.nn as nn


# ============================================================
# Encoder
# ============================================================

class TemporalEncoder(nn.Module):
    """
    Lightweight temporal encoder:
    Linear -> Conv1D -> GRU
    """
    def __init__(self, d_in=64, d_hidden=64, d_gru=128):
        super().__init__()
        self.d_gru = d_gru

        self.stem = nn.Linear(d_in, d_hidden)
        self.conv = nn.Conv1d(d_hidden, d_hidden, kernel_size=3, padding=1)
        self.gru = nn.GRU(d_hidden, d_gru, batch_first=True)
        self.act = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (B, T, F)
        Returns:
            z: (B, T, d_gru)
        """
        x = self.act(self.stem(x))      # (B, T, d_hidden)
        x = x.transpose(1, 2)           # (B, d_hidden, T)
        x = self.act(self.conv(x))
        x = x.transpose(1, 2)           # (B, T, d_hidden)
        z, _ = self.gru(x)              # (B, T, d_gru)
        return z


# ============================================================
# Masked Reconstruction Task
# ============================================================

class MaskedReconstructionTask(nn.Module):
    """
    Self-supervised masked reconstruction with:
    - Temporal span masking
    - Feature-group masking
    - Feature-normalized loss
    """
    def __init__(self, encoder: TemporalEncoder, d_in=64):
        super().__init__()
        self.encoder = encoder
        d_gru = encoder.d_gru  # 🔒 SINGLE SOURCE OF TRUTH

        # Strong reconstruction head
        self.recon_head = nn.Sequential(
            nn.Linear(d_gru, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, d_in)
        )

        # Feature group definitions
        self.feature_groups = [
            list(range(0, 5)),     # geometry
            list(range(5, 8)),     # dynamics
            list(range(8, 11)),    # regime
            list(range(11, 14)),   # confidence
            list(range(14, 26)),   # prices
            list(range(26, 38)),   # volumes
            list(range(38, 46)),   # trades
            list(range(46, 64)),   # derived
        ]

    def forward(self, x, mask_prob=0.15, mask_span_length=5):
        """
        Args:
            x: (B, T, F)
        Returns:
            loss: scalar
        """
        B, T, F = x.shape
        device = x.device

        # ----------------------------------------------------
        # Temporal span masking (biased to recent timesteps)
        # ----------------------------------------------------
        temporal_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

        for b in range(B):
            num_spans = max(1, int(T * mask_prob / mask_span_length))
            for _ in range(num_spans):
                start = int((torch.rand(1, device=device) ** 2) * (T - mask_span_length))
                temporal_mask[b, start:start + mask_span_length] = True

        # ----------------------------------------------------
        # Feature-group masking
        # ----------------------------------------------------
        feature_mask = torch.zeros(B, T, F, dtype=torch.bool, device=device)

        for b in range(B):
            for t in torch.where(temporal_mask[b])[0]:
                if torch.rand(1, device=device) < 0.3:
                    # correlated masking
                    group_pairs = [
                        (4, 5),  # prices + volumes
                        (1, 2),  # dynamics + regime
                        (6, 1),  # trades + dynamics
                    ]
                    g1, g2 = group_pairs[
                        torch.randint(0, len(group_pairs), (1,), device=device).item()
                    ]
                    feature_mask[b, t, self.feature_groups[g1]] = True
                    feature_mask[b, t, self.feature_groups[g2]] = True
                else:
                    num_groups = torch.randint(1, 4, (1,), device=device).item()
                    groups = torch.randperm(len(self.feature_groups), device=device)[:num_groups]
                    for g in groups:
                        feature_mask[b, t, self.feature_groups[g]] = True

        # ----------------------------------------------------
        # Mask input
        # ----------------------------------------------------
        x_masked = x.clone()
        x_masked[feature_mask] = 0.0

        # ----------------------------------------------------
        # Encode + reconstruct
        # ----------------------------------------------------
        z = self.encoder(x_masked)        # (B, T, d_gru)
        x_recon = self.recon_head(z)      # (B, T, F)

        if feature_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Feature-wise scale normalization
        scale = x.std(dim=(0, 1), keepdim=True) + 1e-8
        diff = (x_recon - x) ** 2 / (scale ** 2)

        loss = diff[feature_mask].mean()
        return loss


# ============================================================
# Progressive Masking Schedule
# ============================================================

class ProgressiveMasking:
    """
    Linear curriculum for masking probability.
    """
    def __init__(self, start_prob=0.15, end_prob=0.40, total_epochs=50):
        self.start_prob = start_prob
        self.end_prob = end_prob
        self.total_epochs = total_epochs

    def get_mask_prob(self, epoch):
        progress = min(1.0, epoch / self.total_epochs)
        return self.start_prob + (self.end_prob - self.start_prob) * progress
