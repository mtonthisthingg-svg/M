"""
Log-return features at multiple horizons.

All features use ONLY data strictly before timestamp t (no look-ahead).
The unit test in tests/test_no_lookahead.py verifies this invariant.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from btc_predictor.config import RETURN_HORIZONS_S
from btc_predictor.data.bar_builder import Bar


def log_return(a: float, b: float) -> float:
    """log(b/a), safe against zero/negative."""
    if a <= 0 or b <= 0:
        return 0.0
    return float(np.log(b / a))


def compute_return_features(
    bars: List[Bar],
    now_ts_s: int,
    horizons_s: List[int] = RETURN_HORIZONS_S,
) -> Dict[str, float]:
    """
    Compute log-return features using CLOSE prices.

    For horizon H, the feature is log(price[now-1] / price[now-H-1])
    i.e. the return over the last H seconds, using only bars strictly
    before now_ts_s.

    Args:
        bars:       List of Bar objects sorted by ts_s ascending.
                    Must all have ts_s < now_ts_s (caller responsibility).
        now_ts_s:   Current epoch second (exclusive upper bound).
        horizons_s: Lookback windows in seconds.

    Returns:
        Dict of {"ret_5s": float, "ret_30s": float, ...}
    """
    # Build ts → close price map
    price_map: Dict[int, float] = {b.ts_s: b.close for b in bars if b.ts_s < now_ts_s}

    if not price_map:
        return {f"ret_{h}s": 0.0 for h in horizons_s}

    # Most recent close price (strictly before now)
    latest_ts = max(price_map.keys())
    latest_px  = price_map[latest_ts]

    features: Dict[str, float] = {}
    for h in horizons_s:
        target_ts = now_ts_s - h - 1   # strictly before now
        # Find the bar at or just before target_ts
        candidates = [ts for ts in price_map if ts <= target_ts]
        if candidates:
            ref_ts = max(candidates)
            ref_px = price_map[ref_ts]
            features[f"ret_{h}s"] = log_return(ref_px, latest_px)
        else:
            features[f"ret_{h}s"] = 0.0

    return features


def compute_acceleration(
    bars: List[Bar],
    now_ts_s: int,
) -> Dict[str, float]:
    """
    Return acceleration: difference of short-term returns.
    ret_accel = ret_5s - ret_5s_lagged_5s
    This captures momentum change (2nd derivative of price).
    """
    price_map: Dict[int, float] = {b.ts_s: b.close for b in bars if b.ts_s < now_ts_s}
    if not price_map or len(price_map) < 11:
        return {"ret_accel": 0.0}

    latest_ts = max(price_map.keys())
    latest_px = price_map[latest_ts]

    def px_at_or_before(target: int) -> float:
        cands = [ts for ts in price_map if ts <= target]
        return price_map[max(cands)] if cands else latest_px

    p_5s   = px_at_or_before(now_ts_s - 5 - 1)
    p_10s  = px_at_or_before(now_ts_s - 10 - 1)
    p_15s  = px_at_or_before(now_ts_s - 15 - 1)

    ret_now  = log_return(p_5s, latest_px)
    ret_lag  = log_return(p_10s, p_5s)

    return {"ret_accel": ret_now - ret_lag}
