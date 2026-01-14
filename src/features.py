"""
Feature engineering for LOB prediction.
CRITICAL: Correlation > MSE, Stability > Spikes
"""
import numpy as np
from numba import jit

@jit(nopython=True)
def compute_features(p, v, dp, dv):
    """
    Compute correlation-optimized features from LOB data.
    
    Args:
        p: (T, 12) - prices [bid0-5, ask0-5]
        v: (T, 12) - volumes [bid0-5, ask0-5]
        dp: (T, 4) - trade prices
        dv: (T, 4) - trade volumes
    
    Returns:
        features: (T, 64) - engineered features
    """
    T = p.shape[0]
    features = np.zeros((T, 64), dtype=np.float32)
    
    # === GEOMETRY LAYER ===
    bid_best = p[:, 0]
    ask_best = p[:, 6]
    mid = (bid_best + ask_best) / 2.0
    spread = ask_best - bid_best
    

    weights = np.array([1.0, 0.5, 0.333, 0.25, 0.2, 0.167], dtype=np.float32)
    
    bid_weighted = np.zeros(T, dtype=np.float32)
    ask_weighted = np.zeros(T, dtype=np.float32)
    
    for i in range(6):
        bid_weighted += v[:, i] * weights[i]
        ask_weighted += v[:, 6+i] * weights[i]
    
    imbalance = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + 1e-8)
    

    bid_vol_curve = (v[:, 0] - v[:, 1]) - (v[:, 1] - v[:, 2])
    ask_vol_curve = (v[:, 6] - v[:, 7]) - (v[:, 7] - v[:, 8])
    
    features[:, 0] = mid
    features[:, 1] = spread
    features[:, 2] = imbalance
    features[:, 3] = bid_vol_curve
    features[:, 4] = ask_vol_curve
    

    delta_mid = np.zeros(T, dtype=np.float32)
    delta_mid[1:] = np.diff(mid)
    
    delta_imbalance = np.zeros(T, dtype=np.float32)
    delta_imbalance[1:] = np.diff(imbalance)
    

    signed_pressure = np.zeros(T, dtype=np.float32)
    for t in range(T):
        for i in range(4):
            if dv[t, i] > 0:
                sign = 1.0 if dp[t, i] > mid[t] else -1.0
                signed_pressure[t] += sign * dv[t, i]
    
    features[:, 5] = delta_mid
    features[:, 6] = delta_imbalance
    features[:, 7] = signed_pressure
    

    vol_proxy = np.zeros(T, dtype=np.float32)
    for t in range(10, T):
        vol_proxy[t] = np.std(delta_mid[t-10:t])
    
    # Volume intensity
    total_vol = v[:, :6].sum(axis=1) + v[:, 6:].sum(axis=1)
    vol_intensity = total_vol / (total_vol.mean() + 1e-8)
    

    imb_persist = np.zeros(T, dtype=np.float32)
    for t in range(10, T):
        imb_persist[t] = np.abs(delta_imbalance[t-10:t]).mean()
    
    features[:, 8] = vol_proxy
    features[:, 9] = vol_intensity
    features[:, 10] = imb_persist
    

    vol_threshold = np.zeros(T, dtype=np.float32)
    for t in range(20, T):
        vol_threshold[t] = vol_proxy[t-20:t].mean() + vol_proxy[t-20:t].std()
    vol_high = (vol_proxy > vol_threshold).astype(np.float32)
    

    abs_delta_imb = np.abs(delta_imbalance)
    
    # Trade dominance (volume-weighted)
    trade_dom = np.abs(signed_pressure) / (dv.sum(axis=1) + 1e-8)
    
    features[:, 11] = vol_high
    features[:, 12] = abs_delta_imb
    features[:, 13] = trade_dom
    
    # === RAW FEATURES ===
    features[:, 14:26] = p
    features[:, 26:38] = v
    features[:, 38:42] = dp
    features[:, 42:46] = dv
    
    # Additional derived features for encoder richness
    # Spread dynamics
    features[:, 46] = spread / (spread.mean() + 1e-8)
    
    # Mid-relative prices
    for i in range(6):
        features[:, 47+i] = (p[:, i] - mid) / (spread + 1e-8)  # bid distances
        features[:, 53+i] = (p[:, 6+i] - mid) / (spread + 1e-8)  # ask distances
    
    # REDUCED NORMALIZATION (CRITICAL)
    # Z-score ONLY: imbalance, delta_mid, delta_imbalance
    # indices: 2, 5, 6
    for i in [2, 5, 6]:
        if features[:, i].std() > 1e-6:
            features[:, i] = (features[:, i] - features[:, i].mean()) / features[:, i].std()
    
    # NEVER z-score: volatility, trade pressure, curvature
    # These are already in meaningful units
    
    return features


@jit(nopython=True)
def compute_features_stateful(p, v, dp, dv, history, step):
    """
    Stateful feature computation for inference with rolling context.
    
    Args:
        p, v, dp, dv: current timestep features
        history: dict with rolling windows
        step: current step in sequence
    
    Returns:
        features: (64,) for current timestep
    """
    features = np.zeros(64, dtype=np.float32)
    
    # Current geometry
    mid = (p[0] + p[6]) / 2.0
    spread = p[6] - p[0]
    
    # Update history
    history['mid'][step % 128] = mid
    history['spread'][step % 128] = spread
    
    # Depth-weighted imbalance
    weights = np.array([1.0, 0.5, 0.333, 0.25, 0.2, 0.167], dtype=np.float32)
    bid_weighted = np.sum(v[:6] * weights)
    ask_weighted = np.sum(v[6:] * weights)
    imbalance = (bid_weighted - ask_weighted) / (bid_weighted + ask_weighted + 1e-8)
    history['imbalance'][step % 128] = imbalance
    
    # Volume curvature
    bid_vol_curve = (v[0] - v[1]) - (v[1] - v[2])
    ask_vol_curve = (v[6] - v[7]) - (v[7] - v[8])
    
    features[0] = mid
    features[1] = spread
    features[2] = imbalance
    features[3] = bid_vol_curve
    features[4] = ask_vol_curve
    
    # Dynamics (need history)
    if step > 0:
        prev_mid = history['mid'][(step-1) % 128]
        prev_imb = history['imbalance'][(step-1) % 128]
        delta_mid = mid - prev_mid
        delta_imbalance = imbalance - prev_imb
    else:
        delta_mid = 0.0
        delta_imbalance = 0.0
    
    history['delta_mid'][step % 128] = delta_mid
    history['delta_imbalance'][step % 128] = delta_imbalance
    
    # Signed trade pressure
    signed_pressure = 0.0
    for i in range(4):
        if dv[i] > 0:
            sign = 1.0 if dp[i] > mid else -1.0
            signed_pressure += sign * dv[i]
    
    features[5] = delta_mid
    features[6] = delta_imbalance
    features[7] = signed_pressure
    
    # Regime (need rolling window)
    if step >= 10:
        window_size = min(10, step + 1)
        start_idx = max(0, step - window_size + 1)
        
        # Volatility of delta_mid
        vol_proxy = 0.0
        sum_sq = 0.0
        count = 0
        for i in range(start_idx, step + 1):
            val = history['delta_mid'][i % 128]
            sum_sq += val * val
            count += 1
        vol_proxy = np.sqrt(sum_sq / count) if count > 0 else 0.0
        
        # Imbalance persistence
        imb_persist = 0.0
        for i in range(start_idx, step + 1):
            imb_persist += abs(history['delta_imbalance'][i % 128])
        imb_persist /= count if count > 0 else 1.0
    else:
        vol_proxy = 0.0
        imb_persist = 0.0
    
    total_vol = v[:6].sum() + v[6:].sum()
    vol_intensity = total_vol / (history['vol_mean'] + 1e-8)
    history['vol_sum'] += total_vol
    history['vol_mean'] = history['vol_sum'] / (step + 1)
    
    features[8] = vol_proxy
    features[9] = vol_intensity
    features[10] = imb_persist
    
    # Confidence
    if step >= 20:
        # Rolling volatility threshold
        vol_sum = 0.0
        vol_sq_sum = 0.0
        for i in range(max(0, step - 20), step):
            val = history['vol_hist'][i % 128]
            vol_sum += val
            vol_sq_sum += val * val
        vol_mean = vol_sum / 20
        vol_std = np.sqrt(vol_sq_sum / 20 - vol_mean * vol_mean)
        vol_threshold = vol_mean + vol_std
        vol_high = 1.0 if vol_proxy > vol_threshold else 0.0
    else:
        vol_high = 0.0
    
    history['vol_hist'][step % 128] = vol_proxy
    
    abs_delta_imb = abs(delta_imbalance)
    trade_dom = abs(signed_pressure) / (dv.sum() + 1e-8)
    
    features[11] = vol_high
    features[12] = abs_delta_imb
    features[13] = trade_dom
    
    # Raw features
    features[14:26] = p
    features[26:38] = v
    features[38:42] = dp
    features[42:46] = dv
    
    # Additional features
    features[46] = spread / (history['spread_mean'] + 1e-8)
    history['spread_sum'] += spread
    history['spread_mean'] = history['spread_sum'] / (step + 1)
    
    for i in range(6):
        features[47+i] = (p[i] - mid) / (spread + 1e-8)
        features[53+i] = (p[6+i] - mid) / (spread + 1e-8)
    
    # Normalize ONLY the specified features
    # For inference, use running statistics stored in history
    if step > 10:
        for i in [2, 5, 6]:
            features[i] = (features[i] - history['feature_mean'][i]) / (history['feature_std'][i] + 1e-8)
            
            # Update running stats
            history['feature_sum'][i] += features[i]
            history['feature_sq_sum'][i] += features[i] * features[i]
            n = step + 1
            history['feature_mean'][i] = history['feature_sum'][i] / n
            history['feature_std'][i] = np.sqrt(history['feature_sq_sum'][i] / n - history['feature_mean'][i] ** 2)
    
    return features