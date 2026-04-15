"""
Realized volatility and related dispersion features.

All features use only data strictly before timestamp t.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from btc_predictor.config import REALIZED_VOL_WINDOW_S
from btc_predictor.data.bar_builder import Bar


def realized_vol(
    bars: List[Bar],
    now_ts_s: int,
    window_s: int = REALIZED_VOL_WINDOW_S,
) -> float:
    """
    Annualized realized volatility from 1-second log returns.

    σ = std(log_returns) * sqrt(86400 * 365)   (annualized)

    Only bars with ts_s in [now_ts_s - window_s, now_ts_s) are used.
    """
    start = now_ts_s - window_s
    prices = [b.close for b in bars if start <= b.ts_s < now_ts_s and b.close > 0]
    if len(prices) < 2:
        return 0.0

    rets = np.diff(np.log(prices))
    return float(np.std(rets) * math.sqrt(86400 * 365))


def vol_of_vol(
    bars: List[Bar],
    now_ts_s: int,
    outer_window_s: int = 180,
    inner_window_s: int = 30,
) -> float:
    """
    Volatility-of-volatility: std of rolling 30s realized vols over 180s.
    Captures vol regime changes.
    """
    vols = []
    for offset in range(0, outer_window_s - inner_window_s, inner_window_s):
        end   = now_ts_s - offset
        start = end - inner_window_s
        prices = [b.close for b in bars if start <= b.ts_s < end and b.close > 0]
        if len(prices) >= 2:
            rets = np.diff(np.log(prices))
            vols.append(float(np.std(rets)))

    if len(vols) < 2:
        return 0.0
    return float(np.std(vols))


def vol_ratio(
    bars: List[Bar],
    now_ts_s: int,
    short_s: int = 30,
    long_s: int = 120,
) -> float:
    """
    Short/long realized vol ratio. >1 = vol expansion, <1 = contraction.
    """
    v_short = realized_vol(bars, now_ts_s, window_s=short_s)
    v_long  = realized_vol(bars, now_ts_s, window_s=long_s)
    if v_long > 0:
        return v_short / v_long
    return 1.0


def compute_volatility_features(
    bars: List[Bar],
    now_ts_s: int,
) -> Dict[str, float]:
    return {
        "rvol_30s":   realized_vol(bars, now_ts_s, window_s=30),
        "rvol_60s":   realized_vol(bars, now_ts_s, window_s=60),
        "rvol_180s":  realized_vol(bars, now_ts_s, window_s=180),
        "vol_of_vol": vol_of_vol(bars, now_ts_s),
        "vol_ratio":  vol_ratio(bars, now_ts_s),
    }
