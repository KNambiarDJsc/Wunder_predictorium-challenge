"""
Feature engineering for LOB prediction.
CRITICAL: Correlation > MSE, Stability > Spikes
"""

import numpy as np
from numba import jit



DEPTH_WEIGHTS = np.array(
    [1.0, 0.5, 0.333, 0.25, 0.2, 0.167], dtype=np.float32
)

EPS = 1e-8




@jit(nopython=True)
def compute_features(p, v, dp, dv):
    """
    Args:
        p: (T, 12)
        v: (T, 12)
        dp: (T, 4)
        dv: (T, 4)

    Returns:
        features: (T, 64)
    """
    T = p.shape[0]
    features = np.zeros((T, 64), dtype=np.float32)


    bid_best = p[:, 0]
    ask_best = p[:, 6]
    mid = (bid_best + ask_best) * 0.5
    spread = ask_best - bid_best

    bid_weighted = np.zeros(T, dtype=np.float32)
    ask_weighted = np.zeros(T, dtype=np.float32)

    for i in range(6):
        bid_weighted += v[:, i] * DEPTH_WEIGHTS[i]
        ask_weighted += v[:, 6 + i] * DEPTH_WEIGHTS[i]

    imbalance = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + EPS)
    imbalance = np.clip(imbalance, -1.5, 1.5)

    bid_vol_curve = (v[:, 0] - v[:, 1]) - (v[:, 1] - v[:, 2])
    ask_vol_curve = (v[:, 6] - v[:, 7]) - (v[:, 7] - v[:, 8])

    features[:, 0] = mid
    features[:, 1] = spread
    features[:, 2] = imbalance
    features[:, 3] = bid_vol_curve
    features[:, 4] = ask_vol_curve


    delta_mid = np.zeros(T, dtype=np.float32)
    delta_mid[1:] = mid[1:] - mid[:-1]

    delta_imb = np.zeros(T, dtype=np.float32)
    delta_imb[1:] = imbalance[1:] - imbalance[:-1]

    signed_pressure = np.zeros(T, dtype=np.float32)
    for t in range(T):
        for i in range(4):
            if dv[t, i] > 0:
                signed_pressure[t] += (
                    dv[t, i] if dp[t, i] > mid[t] else -dv[t, i]
                )

    features[:, 5] = delta_mid
    features[:, 6] = delta_imb
    features[:, 7] = signed_pressure


    vol_proxy = np.zeros(T, dtype=np.float32)
    for t in range(10, T):
        vol_proxy[t] = np.std(delta_mid[t - 10 : t])

    total_vol = v[:, :6].sum(axis=1) + v[:, 6:].sum(axis=1)
    vol_intensity = total_vol / (total_vol.mean() + EPS)

    imb_persist = np.zeros(T, dtype=np.float32)
    for t in range(10, T):
        imb_persist[t] = np.abs(delta_imb[t - 10 : t]).mean()

    features[:, 8] = vol_proxy
    features[:, 9] = vol_intensity
    features[:, 10] = imb_persist


    vol_high = np.zeros(T, dtype=np.float32)
    for t in range(20, T):
        threshold = np.percentile(vol_proxy[:t], 75)
        vol_high[t] = 1.0 if vol_proxy[t] > threshold else 0.0

    trade_dom = np.abs(signed_pressure) / (dv.sum(axis=1) + 1.0)

    features[:, 11] = vol_high
    features[:, 12] = np.abs(delta_imb)
    features[:, 13] = trade_dom


    features[:, 14:26] = p
    features[:, 26:38] = v
    features[:, 38:42] = dp
    features[:, 42:46] = dv


    features[:, 46] = spread / (spread.mean() + EPS)

    for i in range(6):
        features[:, 47 + i] = (p[:, i] - mid) / (spread + EPS)
        features[:, 53 + i] = (p[:, 6 + i] - mid) / (spread + EPS)


    for i in (2, 5, 6):
        std = features[:, i].std()
        if std > 1e-6:
            features[:, i] = (features[:, i] - features[:, i].mean()) / std

    return features




@jit(nopython=True)
def compute_features_stateful(p, v, dp, dv, history, step):
    features = np.zeros(64, dtype=np.float32)

    mid = (p[0] + p[6]) * 0.5
    spread = p[6] - p[0]

    history["mid"][step % 128] = mid
    history["spread"][step % 128] = spread

    bid_weighted = np.sum(v[:6] * DEPTH_WEIGHTS)
    ask_weighted = np.sum(v[6:] * DEPTH_WEIGHTS)
    imbalance = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + EPS)
    imbalance = np.clip(imbalance, -1.5, 1.5)

    history["imbalance"][step % 128] = imbalance

    features[0] = mid
    features[1] = spread
    features[2] = imbalance
    features[3] = (v[0] - v[1]) - (v[1] - v[2])
    features[4] = (v[6] - v[7]) - (v[7] - v[8])

    if step > 0:
        delta_mid = mid - history["mid"][(step - 1) % 128]
        delta_imb = imbalance - history["imbalance"][(step - 1) % 128]
    else:
        delta_mid = 0.0
        delta_imb = 0.0

    history["delta_mid"][step % 128] = delta_mid
    history["delta_imb"][step % 128] = delta_imb

    signed_pressure = 0.0
    for i in range(4):
        if dv[i] > 0:
            signed_pressure += dv[i] if dp[i] > mid else -dv[i]

    features[5] = delta_mid
    features[6] = delta_imb
    features[7] = signed_pressure

    if step >= 10:
        vol = 0.0
        imb = 0.0
        for i in range(step - 9, step + 1):
            vol += history["delta_mid"][i % 128] ** 2
            imb += abs(history["delta_imb"][i % 128])
        features[8] = np.sqrt(vol / 10.0)
        features[10] = imb / 10.0
    else:
        features[8] = 0.0
        features[10] = 0.0

    total_vol = v[:6].sum() + v[6:].sum()
    history["vol_sum"] += total_vol
    history["vol_mean"] = history["vol_sum"] / (step + 1)

    features[9] = total_vol / (history["vol_mean"] + EPS)

    history["vol_hist"][step % 128] = features[8]

    if step >= 20:
        threshold = np.percentile(history["vol_hist"][:step], 75)
        features[11] = 1.0 if features[8] > threshold else 0.0
    else:
        features[11] = 0.0

    features[12] = abs(delta_imb)
    features[13] = abs(signed_pressure) / (dv.sum() + 1.0)

    features[14:26] = p
    features[26:38] = v
    features[38:42] = dp
    features[42:46] = dv

    history["spread_sum"] += spread
    history["spread_mean"] = history["spread_sum"] / (step + 1)
    features[46] = spread / (history["spread_mean"] + EPS)

    for i in range(6):
        features[47 + i] = (p[i] - mid) / (spread + EPS)
        features[53 + i] = (p[6 + i] - mid) / (spread + EPS)

    # update stats first, then normalize
    for i in (2, 5, 6):
        history["feature_sum"][i] += features[i]
        history["feature_sq_sum"][i] += features[i] * features[i]
        n = step + 1
        mean = history["feature_sum"][i] / n
        std = np.sqrt(history["feature_sq_sum"][i] / n - mean * mean)
        history["feature_mean"][i] = mean
        history["feature_std"][i] = std

        if step > 10:
            features[i] = (features[i] - mean) / (std + EPS)

    return features
