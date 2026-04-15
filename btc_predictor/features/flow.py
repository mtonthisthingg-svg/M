"""
Trade flow imbalance and VPIN features.

Trade flow imbalance (TFI): signed order flow pressure.
VPIN: Volume-synchronized Probability of Informed Trading
      (Easley, Lopez de Prado, O'Hara 2012).

All features use only data strictly before timestamp t.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List

import numpy as np

from btc_predictor.config import VPIN_BUCKET_SIZE
from btc_predictor.data.bar_builder import Bar


def trade_flow_imbalance(
    bars: List[Bar],
    now_ts_s: int,
    window_s: int = 30,
) -> float:
    """
    TFI = (buy_vol - sell_vol) / total_vol  over window.
    Range [-1, 1].  +1 = all buys (aggressive buyers dominating).

    Uses bar.buy_volume and bar.sell_volume which are trade-classified
    (taker side) in the exchange feed.
    """
    start = now_ts_s - window_s
    relevant = [b for b in bars if start <= b.ts_s < now_ts_s]
    if not relevant:
        return 0.0

    buy_vol  = sum(b.buy_volume  for b in relevant)
    sell_vol = sum(b.sell_volume for b in relevant)
    total    = buy_vol + sell_vol

    if total == 0:
        return 0.0
    return float((buy_vol - sell_vol) / total)


def volume_momentum(
    bars: List[Bar],
    now_ts_s: int,
    short_window_s: int = 10,
    long_window_s: int = 60,
) -> float:
    """
    Volume momentum: ratio of recent volume to baseline volume.
    >1 = elevated activity (potential informed flow).
    """
    short_start = now_ts_s - short_window_s
    long_start  = now_ts_s - long_window_s

    short_vol = sum(b.volume for b in bars if short_start <= b.ts_s < now_ts_s)
    long_vol  = sum(b.volume for b in bars if long_start  <= b.ts_s < now_ts_s)

    long_avg = long_vol / (long_window_s / short_window_s) if long_vol > 0 else 0
    if long_avg == 0:
        return 1.0
    return float(short_vol / long_avg)


class VPINCalculator:
    """
    Rolling VPIN (Volume-synchronized Probability of Informed Trading).

    Accumulates trades into equal-volume buckets; VPIN is the average
    absolute order imbalance across the last N buckets.

    Reference: Easley, Lopez de Prado & O'Hara (2012).
    """

    def __init__(self, bucket_size: float = VPIN_BUCKET_SIZE, n_buckets: int = 50) -> None:
        self._bucket_size = bucket_size
        self._n_buckets = n_buckets
        self._current_buy_vol  = 0.0
        self._current_sell_vol = 0.0
        self._current_total    = 0.0
        # Deque of (buy_vol, sell_vol) per completed bucket
        self._buckets: deque[tuple[float, float]] = deque(maxlen=n_buckets)

    def update(self, buy_vol: float, sell_vol: float) -> None:
        """Add one bar's buy/sell volumes."""
        remaining = self._bucket_size - self._current_total
        to_add = min(buy_vol + sell_vol, remaining)
        ratio = (buy_vol / (buy_vol + sell_vol)) if (buy_vol + sell_vol) > 0 else 0.5

        self._current_buy_vol  += to_add * ratio
        self._current_sell_vol += to_add * (1 - ratio)
        self._current_total    += to_add

        if self._current_total >= self._bucket_size:
            self._buckets.append((self._current_buy_vol, self._current_sell_vol))
            self._current_buy_vol  = 0.0
            self._current_sell_vol = 0.0
            self._current_total    = 0.0

    @property
    def vpin(self) -> float:
        """Current VPIN estimate. Range [0, 1]. Higher = more informed flow."""
        if len(self._buckets) < 2:
            return 0.0
        imbalances = [
            abs(b - s) / (b + s) if (b + s) > 0 else 0.0
            for b, s in self._buckets
        ]
        return float(np.mean(imbalances))


def compute_flow_features(
    bars: List[Bar],
    now_ts_s: int,
    vpin_calc: VPINCalculator,
) -> Dict[str, float]:
    """
    Compute all flow features.

    Args:
        bars:       1-second bars (ts_s < now_ts_s).
        now_ts_s:   Current epoch second.
        vpin_calc:  Stateful VPIN calculator — caller updates it externally.
    """
    return {
        "tfi_10s":      trade_flow_imbalance(bars, now_ts_s, window_s=10),
        "tfi_30s":      trade_flow_imbalance(bars, now_ts_s, window_s=30),
        "tfi_60s":      trade_flow_imbalance(bars, now_ts_s, window_s=60),
        "vol_momentum": volume_momentum(bars, now_ts_s),
        "vpin":         vpin_calc.vpin,
    }
