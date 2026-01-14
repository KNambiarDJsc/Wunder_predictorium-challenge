"""
Correlation-aware loss functions.
EXTREMELY IMPORTANT: Proper Pearson computation, no zero-mask issues
"""
import torch
import torch.nn as nn


def weighted_pearson_loss(pred, target, weights):
    """
    Weighted Pearson correlation loss.
    Loss = 1 - corr
    """
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    w_sum = weights.sum()
    if w_sum < 1e-8:
        return torch.tensor(0.0, device=pred.device)

    # Weighted means
    pred_mean = (pred * weights).sum() / w_sum
    target_mean = (target * weights).sum() / w_sum

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Weighted covariance
    cov = (weights * pred_centered * target_centered).sum() / w_sum

    # Weighted variances
    var_pred = (weights * pred_centered**2).sum() / w_sum
    var_target = (weights * target_centered**2).sum() / w_sum

    if var_pred < 1e-8 or var_target < 1e-8:
        return torch.tensor(1.0, device=pred.device)

    corr = cov / (torch.sqrt(var_pred * var_target) + 1e-8)

    return 1.0 - corr


def weighted_mse_loss(pred, target, weights):
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    return ((pred - target) ** 2 * weights).sum() / (weights.sum() + 1e-8)


def confidence_penalty_v2(conf, target):
    """
    Penalize confidence ONLY when |target| is small.
    """
    if conf.numel() == 0:
        return torch.tensor(0.0, device=conf.device)

    small_target_threshold = 0.5
    is_small = (target.abs() < small_target_threshold).float()

    return (conf * is_small).mean()


def abstention_reward(conf, target):
    """
    Soft abstention encouragement:
    reward = conf * exp(-|target|)
    """
    if conf.numel() == 0:
        return torch.tensor(0.0, device=conf.device)

    return (conf * torch.exp(-target.abs())).mean()


class CompetitionLoss(nn.Module):
    """
    Combined loss for LOB prediction competition.
    """
    def __init__(self, alpha_mse=0.15, alpha_conf=0.1, alpha_abstain=0.05):
        super().__init__()
        self.alpha_mse = alpha_mse
        self.alpha_conf = alpha_conf
        self.alpha_abstain = alpha_abstain

    def forward(self, pred, conf, target, pred_mask):
        # Handle (B,T,2) or (B,2)
        if pred.dim() == 3:
            B, T, _ = pred.shape

            mask = pred_mask.reshape(-1)
            pred = pred.reshape(-1, 2)[mask]
            target = target.reshape(-1, 2)[mask]
            conf = conf.reshape(-1)[mask]
        else:
            conf = conf.squeeze(-1)

        if pred.numel() == 0:
            zero = torch.tensor(0.0, device=pred.device)
            return zero, {
                "corr_t0": 0.0,
                "corr_t1": 0.0,
                "mse": 0.0,
                "conf_penalty": 0.0,
                "abstain_reward": 0.0,
                "mean_conf": 0.0,
            }

        pred_t0, pred_t1 = pred[:, 0], pred[:, 1]
        target_t0, target_t1 = target[:, 0], target[:, 1]

        weights_t0 = target_t0.abs() + 0.1
        weights_t1 = target_t1.abs() + 0.1

        loss_corr_t0 = weighted_pearson_loss(pred_t0, target_t0, weights_t0)
        loss_corr_t1 = weighted_pearson_loss(pred_t1, target_t1, weights_t1)
        loss_corr = 0.5 * (loss_corr_t0 + loss_corr_t1)

        loss_mse = weighted_mse_loss(
            torch.cat([pred_t0, pred_t1]),
            torch.cat([target_t0, target_t1]),
            torch.cat([weights_t0, weights_t1]),
        )

        # Confidence penalties (ONCE per timestep)
        loss_conf = 0.5 * (
            confidence_penalty_v2(conf, target_t0) +
            confidence_penalty_v2(conf, target_t1)
        )

        loss_abstain = 0.5 * (
            abstention_reward(conf, target_t0) +
            abstention_reward(conf, target_t1)
        )

        loss = (
            loss_corr
            + self.alpha_mse * loss_mse
            + self.alpha_conf * loss_conf
            + self.alpha_abstain * loss_abstain
        )

        metrics = {
            "corr_t0": 1.0 - loss_corr_t0.item(),
            "corr_t1": 1.0 - loss_corr_t1.item(),
            "mse": loss_mse.item(),
            "conf_penalty": loss_conf.item(),
            "abstain_reward": loss_abstain.item(),
            "mean_conf": conf.mean().item(),
        }

        return loss, metrics
