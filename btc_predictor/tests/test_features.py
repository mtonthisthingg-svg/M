"""
Feature pipeline sanity tests.

Verifies:
  1. All features are finite (no NaN/inf)
  2. Feature names are consistent across calls
  3. Key features are in expected ranges
  4. OBI is in [-1, 1]
  5. VPIN is in [0, 1]
  6. Calibration brier score improves after isotonic fit
"""

from __future__ import annotations

import numpy as np
import pytest

from btc_predictor.data.bar_builder import Bar
from btc_predictor.features.orderbook import compute_orderbook_features
from btc_predictor.features.flow import VPINCalculator, compute_flow_features
from btc_predictor.features.returns import compute_return_features
from btc_predictor.features.volatility import compute_volatility_features
from btc_predictor.models.calibration import IsotonicCalibrator


def make_bar(ts_s: int, price: float = 50_000.0) -> Bar:
    b = Bar(
        ts_s=ts_s, exchange="test",
        open=price, high=price+10, low=price-10, close=price,
        volume=1.0, buy_volume=0.6, sell_volume=0.4, trade_count=10,
    )
    b.bid_prices = np.array([price - 1.0, price - 2.0, price - 3.0,
                              price - 4.0, price - 5.0] * 4, dtype=float)[:20]
    b.bid_sizes  = np.ones(20)
    b.ask_prices = np.array([price + 1.0, price + 2.0, price + 3.0,
                              price + 4.0, price + 5.0] * 4, dtype=float)[:20]
    b.ask_sizes  = np.ones(20)
    b.midprice   = price
    b.microprice = price
    b.spread     = 2.0
    return b


class TestReturnFeatures:
    def test_all_finite(self):
        bars = [make_bar(i) for i in range(200)]
        feats = compute_return_features(bars, now_ts_s=100)
        for k, v in feats.items():
            assert np.isfinite(v), f"{k}={v} is not finite"

    def test_consistent_keys(self):
        bars = [make_bar(i) for i in range(200)]
        feats1 = compute_return_features(bars, now_ts_s=50)
        feats2 = compute_return_features(bars, now_ts_s=100)
        assert set(feats1.keys()) == set(feats2.keys())

    def test_empty_returns_zeros(self):
        feats = compute_return_features([], now_ts_s=100)
        for v in feats.values():
            assert v == 0.0


class TestOrderbookFeatures:
    def test_obi_in_range(self):
        bar = make_bar(99)
        feats = compute_orderbook_features(bar)
        assert -1.0 <= feats["obi_shallow"] <= 1.0
        assert -1.0 <= feats["obi_deep"] <= 1.0

    def test_balanced_book_obi_near_zero(self):
        """Equal bid/ask sizes → OBI ≈ 0."""
        bar = make_bar(99)
        feats = compute_orderbook_features(bar)
        assert abs(feats["obi_shallow"]) < 0.1

    def test_none_returns_zeros(self):
        feats = compute_orderbook_features(None)
        assert feats["obi_shallow"] == 0.0
        assert feats["spread_bps"] == 0.0

    def test_spread_non_negative(self):
        bar = make_bar(99)
        feats = compute_orderbook_features(bar)
        assert feats["spread_bps"] >= 0

    def test_all_finite(self):
        bar = make_bar(99)
        feats = compute_orderbook_features(bar)
        for k, v in feats.items():
            assert np.isfinite(v), f"{k}={v}"


class TestFlowFeatures:
    def test_tfi_in_range(self):
        bars = [make_bar(i) for i in range(100)]
        vpin = VPINCalculator()
        feats = compute_flow_features(bars, now_ts_s=50, vpin_calc=vpin)
        for k in ["tfi_10s", "tfi_30s", "tfi_60s"]:
            assert -1.0 <= feats[k] <= 1.0, f"{k}={feats[k]}"

    def test_vpin_in_range(self):
        vpin = VPINCalculator(bucket_size=10)
        for i in range(200):
            vpin.update(buy_vol=0.6, sell_vol=0.4)
        assert 0.0 <= vpin.vpin <= 1.0

    def test_all_buys_tfi_positive(self):
        """All taker buys → TFI should be positive."""
        bars = []
        for i in range(50):
            b = make_bar(i)
            b.buy_volume  = 1.0
            b.sell_volume = 0.0
            bars.append(b)
        vpin = VPINCalculator()
        feats = compute_flow_features(bars, now_ts_s=40, vpin_calc=vpin)
        assert feats["tfi_30s"] > 0


class TestCalibration:
    def test_calibration_reduces_brier(self):
        """Isotonic calibration should reduce or maintain Brier score."""
        rng = np.random.default_rng(42)
        # Generate over-confident predictions (miscalibrated)
        true_labels = rng.integers(0, 2, size=500).astype(float)
        raw_probs   = true_labels * 0.8 + (1 - true_labels) * 0.2  # over-confident
        # Add noise
        raw_probs   = np.clip(raw_probs + rng.normal(0, 0.15, 500), 0.01, 0.99)

        cal = IsotonicCalibrator()
        # Split 50/50
        n = len(raw_probs) // 2
        cal.fit(raw_probs[:n], true_labels[:n])
        cal_probs = cal.transform(raw_probs[n:])

        brier_raw = float(np.mean((raw_probs[n:] - true_labels[n:]) ** 2))
        brier_cal = float(np.mean((cal_probs - true_labels[n:]) ** 2))

        # Calibrated Brier should be ≤ raw Brier (with small tolerance)
        assert brier_cal <= brier_raw + 0.01, (
            f"Calibration made Brier worse: {brier_raw:.4f} → {brier_cal:.4f}"
        )

    def test_reliability_diagram_shapes(self):
        rng = np.random.default_rng(42)
        probs  = rng.uniform(0, 1, 200)
        labels = (probs + rng.normal(0, 0.1, 200) > 0.5).astype(float)
        cal = IsotonicCalibrator()
        mean_pred, frac_pos, counts = cal.reliability_diagram(probs, labels)
        assert len(mean_pred) == len(frac_pos) == len(counts)
        assert all(0 <= m <= 1 for m in mean_pred)
        assert all(0 <= f <= 1 for f in frac_pos)
