"""
Order book imbalance, microprice, spread, and depth features.

All features use only the latest orderbook snapshot strictly before t.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from btc_predictor.config import OBI_LEVELS_DEEP, OBI_LEVELS_SHALLOW
from btc_predictor.data.bar_builder import Bar


def order_book_imbalance(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    n_levels: int,
) -> float:
    """
    OBI = (sum_bid_qty - sum_ask_qty) / (sum_bid_qty + sum_ask_qty)
    Range [-1, 1].  +1 = all bids, -1 = all asks.
    """
    b = np.sum(bid_sizes[:n_levels])
    a = np.sum(ask_sizes[:n_levels])
    denom = b + a
    if denom == 0:
        return 0.0
    return float((b - a) / denom)


def weighted_midprice(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    n_levels: int = 5,
) -> float:
    """
    Volume-weighted midprice over top-N levels.
    More robust than simple midprice in volatile books.
    """
    bp = bid_prices[:n_levels]
    bs = bid_sizes[:n_levels]
    ap = ask_prices[:n_levels]
    as_ = ask_sizes[:n_levels]

    total = np.sum(bs) + np.sum(as_)
    if total == 0:
        return 0.0

    wmid = (np.dot(bp, bs) + np.dot(ap, as_)) / total
    return float(wmid)


def depth_ratio(
    bid_sizes: np.ndarray,
    ask_sizes: np.ndarray,
    shallow_n: int,
    deep_n: int,
) -> float:
    """
    Ratio of shallow depth to deep depth.
    Low ratio = thin book at best levels (fragile).
    """
    shallow = np.sum(bid_sizes[:shallow_n]) + np.sum(ask_sizes[:shallow_n])
    deep    = np.sum(bid_sizes[:deep_n])    + np.sum(ask_sizes[:deep_n])
    if deep == 0:
        return 1.0
    return float(shallow / deep)


def compute_orderbook_features(bar: Optional[Bar]) -> Dict[str, float]:
    """
    Compute orderbook features from the most recent bar's L2 snapshot.

    Args:
        bar: Most recent Bar (ts_s < now_ts_s, caller ensures this).

    Returns:
        Dict of orderbook features.
    """
    if bar is None:
        return _zero_features()

    bp = bar.bid_prices
    bs = bar.bid_sizes
    ap = bar.ask_prices
    as_ = bar.ask_sizes

    if len(bp) == 0 or len(ap) == 0:
        return _zero_features()

    best_bid = bp[0] if len(bp) > 0 else 0.0
    best_ask = ap[0] if len(ap) > 0 else 0.0
    if best_bid == 0 or best_ask == 0:
        return _zero_features()

    spread   = best_ask - best_bid
    mid      = (best_bid + best_ask) / 2.0
    micro    = bar.microprice if bar.microprice > 0 else mid
    micro_dev = (micro - mid) / mid if mid > 0 else 0.0

    obi_shallow = order_book_imbalance(bs, as_, OBI_LEVELS_SHALLOW)
    obi_deep    = order_book_imbalance(bs, as_, OBI_LEVELS_DEEP)
    wmid        = weighted_midprice(bp, bs, ap, as_, n_levels=5)
    wmid_dev    = (wmid - mid) / mid if mid > 0 else 0.0
    depth_r     = depth_ratio(bs, as_, OBI_LEVELS_SHALLOW, OBI_LEVELS_DEEP)

    # Relative spread (basis points)
    spread_bps  = spread / mid * 10_000 if mid > 0 else 0.0

    # Cumulative depth at 5/10/20 bps from mid
    depth_5bps  = _cum_depth_near_mid(bp, bs, ap, as_, bps_from_mid=5,  mid=mid)
    depth_10bps = _cum_depth_near_mid(bp, bs, ap, as_, bps_from_mid=10, mid=mid)

    return {
        "obi_shallow":  obi_shallow,
        "obi_deep":     obi_deep,
        "micro_dev":    micro_dev,
        "wmid_dev":     wmid_dev,
        "spread_bps":   spread_bps,
        "depth_ratio":  depth_r,
        "depth_5bps":   depth_5bps,
        "depth_10bps":  depth_10bps,
    }


def _cum_depth_near_mid(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    bps_from_mid: float,
    mid: float,
) -> float:
    """Sum of bid+ask sizes within `bps_from_mid` basis points of mid."""
    thresh = mid * bps_from_mid / 10_000
    bid_depth = sum(
        s for p, s in zip(bid_prices, bid_sizes)
        if mid - p <= thresh
    )
    ask_depth = sum(
        s for p, s in zip(ask_prices, ask_sizes)
        if p - mid <= thresh
    )
    return float(bid_depth + ask_depth)


def _zero_features() -> Dict[str, float]:
    return {
        "obi_shallow":  0.0,
        "obi_deep":     0.0,
        "micro_dev":    0.0,
        "wmid_dev":     0.0,
        "spread_bps":   0.0,
        "depth_ratio":  1.0,
        "depth_5bps":   0.0,
        "depth_10bps":  0.0,
    }
