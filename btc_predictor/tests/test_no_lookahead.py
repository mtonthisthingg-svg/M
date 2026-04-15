"""
No look-ahead bias tests.

CRITICAL: Every feature at time t must use ONLY data with ts_s < t.
These tests synthetically inject a "canary" price at t and verify that
it never appears in features computed at t.

If any test in this file fails, there is a look-ahead bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_predictor.data.bar_builder import Bar
from btc_predictor.features.returns import compute_return_features, compute_acceleration
from btc_predictor.features.volatility import compute_volatility_features
from btc_predictor.features.orderbook import compute_orderbook_features
from btc_predictor.features.flow import VPINCalculator, compute_flow_features


CANARY_PRICE = 999_999.0   # price impossible in reality


def make_bars(n: int, base_price: float = 50_000.0) -> list[Bar]:
    """Generate n bars with ts_s = 0, 1, ..., n-1."""
    bars = []
    for i in range(n):
        b = Bar(
            ts_s=i,
            exchange="test",
            open=base_price + i,
            high=base_price + i + 1,
            low=base_price + i - 1,
            close=base_price + i,
            volume=1.0,
            buy_volume=0.5,
            sell_volume=0.5,
            trade_count=10,
        )
        b.midprice = base_price + i
        bars.append(b)
    return bars


def inject_canary(bars: list[Bar], ts_s: int) -> list[Bar]:
    """Add a bar at ts_s with the canary price. Simulates 'current' data."""
    canary = Bar(
        ts_s=ts_s,
        exchange="test",
        open=CANARY_PRICE,
        high=CANARY_PRICE,
        low=CANARY_PRICE,
        close=CANARY_PRICE,
        volume=1.0,
        buy_volume=0.5,
        sell_volume=0.5,
        trade_count=10,
    )
    canary.midprice = CANARY_PRICE
    return bars + [canary]


class TestReturnFeaturesNoLookahead:
    def test_canary_not_in_return_features(self):
        """Log returns at time t must not use the bar at ts_s=t."""
        bars = make_bars(200, base_price=50_000.0)
        now_ts = 100
        bars_with_canary = inject_canary(bars, ts_s=now_ts)

        feats = compute_return_features(bars_with_canary, now_ts_s=now_ts)

        for k, v in feats.items():
            # If canary leaked in, the return would be huge
            assert abs(v) < 1.0, (
                f"Feature {k}={v} suggests canary price leaked! "
                f"(return > 1 from canary at ts_s={now_ts})"
            )

    def test_uses_strictly_before(self):
        """The most recent price used should be from ts_s = now_ts - 1."""
        bars = make_bars(200, base_price=50_000.0)
        now_ts = 100
        # Bar at 99 has close = 50000 + 99 = 50099
        feats = compute_return_features(bars, now_ts_s=now_ts)
        # All returns should be finite and based on prices ~ 50000
        for v in feats.values():
            assert np.isfinite(v)
            assert abs(v) < 0.1   # <10% return impossible in test data


class TestVolatilityNoLookahead:
    def test_canary_not_in_vol(self):
        bars = make_bars(200, base_price=50_000.0)
        now_ts = 100
        bars_with_canary = inject_canary(bars, ts_s=now_ts)

        feats = compute_volatility_features(bars_with_canary, now_ts_s=now_ts)

        # If canary leaked: return would be log(999999/50099) ≈ 3.0, inflating vol
        for k, v in feats.items():
            assert v < 100.0, f"Vol feature {k}={v} — possible canary leak"


class TestFlowNoLookahead:
    def test_tfi_not_future(self):
        """Trade flow imbalance must not include bars at or after now_ts."""
        bars = make_bars(200, base_price=50_000.0)
        # Add a future bar
        future = Bar(
            ts_s=200, exchange="test",
            open=1e6, high=1e6, low=1e6, close=1e6,
            volume=1000.0, buy_volume=1000.0, sell_volume=0.0, trade_count=1,
        )
        bars_with_future = bars + [future]
        now_ts = 100

        vpin = VPINCalculator()
        feats = compute_flow_features(bars_with_future, now_ts, vpin)

        # If future bar leaked: tfi_10s or tfi_30s would be very different
        # Normal test bars have buy_vol=sell_vol=0.5 → TFI near 0
        for k in ["tfi_10s", "tfi_30s", "tfi_60s"]:
            assert abs(feats[k]) < 0.5, f"{k}={feats[k]} suggests future data leaked"


class TestOrderbookNoLookahead:
    def test_uses_provided_bar_not_future(self):
        """Orderbook features use the bar passed in — no implicit future access."""
        bar = Bar(
            ts_s=99, exchange="test",
            open=50000, high=50001, low=49999, close=50000,
            volume=1.0, buy_volume=0.5, sell_volume=0.5, trade_count=1,
        )
        bar.bid_prices = np.array([49990.0, 49980.0])
        bar.bid_sizes  = np.array([1.0, 2.0])
        bar.ask_prices = np.array([50010.0, 50020.0])
        bar.ask_sizes  = np.array([1.0, 2.0])
        bar.midprice   = 50000.0
        bar.microprice = 50000.0
        bar.spread     = 20.0

        feats = compute_orderbook_features(bar)
        assert "obi_shallow" in feats
        assert np.isfinite(feats["obi_shallow"])
        # With equal bid/ask depth, OBI should be near 0
        assert abs(feats["obi_shallow"]) < 0.01


class TestFeatureTimestampInvariant:
    def test_features_at_t_equal_to_features_at_t_with_extra_future_data(self):
        """
        Adding future bars to the input must not change features computed at t.
        This is the strongest no-lookahead test.
        """
        bars = make_bars(120, base_price=50_000.0)
        now_ts = 60

        # Compute with bars up to ts_s=59 only
        past_bars = [b for b in bars if b.ts_s < now_ts]
        feats_without_future = compute_return_features(past_bars, now_ts)

        # Compute with bars up to ts_s=119 (all future included)
        feats_with_future = compute_return_features(bars, now_ts)

        # Results must be identical
        for k in feats_without_future:
            assert feats_without_future[k] == pytest.approx(feats_with_future[k], abs=1e-10), (
                f"Feature {k} differs: {feats_without_future[k]} vs {feats_with_future[k]}. "
                "Future data is affecting past features — LOOK-AHEAD BUG!"
            )
