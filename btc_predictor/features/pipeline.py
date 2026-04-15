"""
Main feature engineering pipeline.

Produces a single flat feature vector at time t using only data
strictly available before t (no look-ahead, verified by tests).

Feature groups:
  1. Price returns (multi-horizon log returns + acceleration)
  2. Realized volatility (multi-window + vol-of-vol + vol-ratio)
  3. Orderbook imbalance (shallow + deep OBI, microprice, spread, depth)
  4. Trade flow (TFI, volume momentum, VPIN)
  5. Oracle lead (binance_mid - chainlink_price) — PRIMARY EDGE FEATURE
  6. Polymarket implied probability and book depth
  7. Slow signals (funding rate, CME basis)
  8. Market microstructure (time into window, open price gap)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from btc_predictor.config import (
    MARKET_WINDOW_SECONDS,
    RETURN_HORIZONS_S,
    SEQUENCE_LENGTH_S,
)
from btc_predictor.data.bar_builder import Bar, MultiExchangeBarStore
from btc_predictor.data.chainlink_feed import ChainlinkFeed
from btc_predictor.data.funding_rates import FundingRateCollector
from btc_predictor.data.cme_basis import CMEBasisCollector
from btc_predictor.data.polymarket_feed import PolymarketFeed, PolymarketMarket
from btc_predictor.features.returns import compute_return_features, compute_acceleration
from btc_predictor.features.volatility import compute_volatility_features
from btc_predictor.features.orderbook import compute_orderbook_features
from btc_predictor.features.flow import VPINCalculator, compute_flow_features


FEATURE_NAMES: List[str] = []  # populated on first call to compute_features()


@dataclass
class FeatureVector:
    ts_s: int
    features: Dict[str, float]
    sequence: Optional[np.ndarray] = None  # (SEQUENCE_LENGTH_S, n_features) for CNN


class FeaturePipeline:
    """
    Stateful feature pipeline. Call update() on each new bar, then
    call compute(now_ts_s) to get the current feature vector.

    The pipeline is exchange-aware: it uses Binance as the primary
    price feed (most liquid) and Coinbase/Bybit for cross-venue features.
    """

    PRIMARY_EXCHANGE = "binance"

    def __init__(
        self,
        store: MultiExchangeBarStore,
        chainlink: ChainlinkFeed,
        polymarket: PolymarketFeed,
        funding: Optional[FundingRateCollector] = None,
        cme: Optional[CMEBasisCollector] = None,
    ) -> None:
        self._store      = store
        self._chainlink  = chainlink
        self._polymarket = polymarket
        self._funding    = funding
        self._cme        = cme
        self._vpin_binance  = VPINCalculator()
        self._vpin_coinbase = VPINCalculator()
        # Rolling feature history for CNN sequence
        self._feature_history: List[Dict[str, float]] = []
        self._history_max = SEQUENCE_LENGTH_S + 10

    async def compute(self, now_ts_s: int) -> FeatureVector:
        """
        Compute all features using data strictly before now_ts_s.
        This is the main entry point called every second.
        """
        # ----------------------------------------------------------------
        # Fetch bar windows
        # ----------------------------------------------------------------
        primary_bars = await self._store.get_window(
            self.PRIMARY_EXCHANGE, now_ts_s, lookback_s=max(RETURN_HORIZONS_S) + 10
        )
        coinbase_bars = await self._store.get_window(
            "coinbase", now_ts_s, lookback_s=60
        )

        latest_primary = primary_bars[-1] if primary_bars else None
        latest_coinbase = coinbase_bars[-1] if coinbase_bars else None

        # ----------------------------------------------------------------
        # Update VPIN calculators (stateful)
        # ----------------------------------------------------------------
        for bar in primary_bars:
            self._vpin_binance.update(bar.buy_volume, bar.sell_volume)
        for bar in coinbase_bars:
            self._vpin_coinbase.update(bar.buy_volume, bar.sell_volume)

        # ----------------------------------------------------------------
        # Oracle lead: THE primary edge feature
        # Chainlink BTC/USD lags Binance spot by 100ms-2s.
        # When Binance has moved significantly above/below Chainlink,
        # it predicts the direction Chainlink (= resolution oracle) will move.
        # ----------------------------------------------------------------
        cl_ts_ms, cl_price = self._chainlink.latest
        binance_mid = latest_primary.midprice if latest_primary else 0.0
        oracle_lead = 0.0
        oracle_lead_bps = 0.0
        if cl_price > 0 and binance_mid > 0:
            oracle_lead = binance_mid - cl_price
            oracle_lead_bps = oracle_lead / cl_price * 10_000

        # ----------------------------------------------------------------
        # Price return features (primary exchange)
        # ----------------------------------------------------------------
        ret_feats = compute_return_features(primary_bars, now_ts_s)
        accel_feats = compute_acceleration(primary_bars, now_ts_s)

        # ----------------------------------------------------------------
        # Volatility features
        # ----------------------------------------------------------------
        vol_feats = compute_volatility_features(primary_bars, now_ts_s)

        # ----------------------------------------------------------------
        # Orderbook features (primary + Coinbase for cross-venue OBI)
        # ----------------------------------------------------------------
        ob_feats_binance  = compute_orderbook_features(latest_primary)
        ob_feats_coinbase = compute_orderbook_features(latest_coinbase)

        # Cross-venue OBI spread: divergence signals informed flow on one venue
        cross_obi = (
            ob_feats_binance["obi_shallow"] - ob_feats_coinbase["obi_shallow"]
        )

        # ----------------------------------------------------------------
        # Flow features
        # ----------------------------------------------------------------
        flow_feats = compute_flow_features(primary_bars, now_ts_s, self._vpin_binance)

        # ----------------------------------------------------------------
        # Polymarket features
        # ----------------------------------------------------------------
        pm_market = self._polymarket.current_market
        pm_feats = _polymarket_features(pm_market, now_ts_s, cl_price)

        # ----------------------------------------------------------------
        # Slow signals
        # ----------------------------------------------------------------
        funding_feats = _funding_features(self._funding)
        cme_feats     = _cme_features(self._cme)

        # ----------------------------------------------------------------
        # Assemble
        # ----------------------------------------------------------------
        features: Dict[str, float] = {}
        features.update(ret_feats)
        features.update(accel_feats)
        features.update(vol_feats)
        features.update({f"bnb_{k}": v for k, v in ob_feats_binance.items()})
        features.update({f"cb_{k}":  v for k, v in ob_feats_coinbase.items()})
        features["cross_obi"] = cross_obi
        features.update(flow_feats)
        features["oracle_lead"]     = oracle_lead
        features["oracle_lead_bps"] = oracle_lead_bps
        features.update(pm_feats)
        features.update(funding_feats)
        features.update(cme_feats)

        # Ensure no NaN/inf
        for k in features:
            v = features[k]
            if not math.isfinite(v):
                features[k] = 0.0

        # Store in history for CNN
        self._feature_history.append(features)
        if len(self._feature_history) > self._history_max:
            self._feature_history.pop(0)

        # Build sequence array (CNN input)
        seq = _build_sequence(self._feature_history, SEQUENCE_LENGTH_S)

        return FeatureVector(ts_s=now_ts_s, features=features, sequence=seq)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _polymarket_features(
    market: Optional[PolymarketMarket],
    now_ts_s: int,
    chainlink_price: float,
) -> Dict[str, float]:
    if market is None:
        return {
            "pm_implied_up":      0.5,
            "pm_spread":          1.0,
            "pm_time_remaining":  0.0,
            "pm_time_frac":       0.5,
            "pm_open_gap_bps":    0.0,
            "pm_book_depth_up":   0.0,
        }

    time_remaining = market.time_remaining_s
    time_frac = 1.0 - (time_remaining / MARKET_WINDOW_SECONDS)

    # Open gap: how far Chainlink has moved from the locked open price
    open_gap_bps = 0.0
    if market.chainlink_open_price > 0 and chainlink_price > 0:
        open_gap_bps = (
            (chainlink_price - market.chainlink_open_price)
            / market.chainlink_open_price * 10_000
        )

    # Orderbook depth on UP side
    up_depth = sum(q for _, q in market.up_asks[:5]) + sum(q for _, q in market.up_bids[:5])

    return {
        "pm_implied_up":      market.mid,
        "pm_spread":          market.spread,
        "pm_time_remaining":  time_remaining,
        "pm_time_frac":       time_frac,
        "pm_open_gap_bps":    open_gap_bps,
        "pm_book_depth_up":   up_depth,
    }


def _funding_features(collector: Optional[FundingRateCollector]) -> Dict[str, float]:
    if collector is None or collector.latest is None:
        return {"funding_binance": 0.0, "funding_bybit": 0.0}
    snap = collector.latest
    return {
        "funding_binance": snap.binance_funding_rate,
        "funding_bybit":   snap.bybit_funding_rate,
    }


def _cme_features(collector: Optional[CMEBasisCollector]) -> Dict[str, float]:
    if collector is None or collector.latest is None:
        return {"cme_basis_ann": 0.0}
    return {"cme_basis_ann": collector.latest.basis_annualized}


def _build_sequence(
    history: List[Dict[str, float]],
    length: int,
) -> Optional[np.ndarray]:
    """
    Build (length, n_features) array from the last `length` feature dicts.
    Returns None if not enough history yet.
    """
    if len(history) < length:
        return None
    window = history[-length:]
    keys = sorted(window[0].keys())
    arr = np.array([[row.get(k, 0.0) for k in keys] for row in window], dtype=np.float32)
    return arr


def feature_names_from(fv: FeatureVector) -> List[str]:
    """Sorted list of feature names (used for consistent column ordering)."""
    return sorted(fv.features.keys())
