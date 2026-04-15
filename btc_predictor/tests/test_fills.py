"""
Fill simulation tests.

Verifies:
  1. PnL calculation is correct (win/loss)
  2. Break-even edge formula is consistent with EV calculation
  3. Kelly sizing is positive only when EV > 0
  4. Partial fills work correctly
  5. Fee accounting is correct
"""

from __future__ import annotations

import pytest

from btc_predictor.backtest.fills import (
    Fill,
    OrderbookLevel,
    break_even_edge,
    resolve_fill,
    simulate_fill,
)
from btc_predictor.live.sizing import expected_value_per_dollar, fractional_kelly


class TestFillSimulation:
    def _make_asks(self, mid: float, n_levels: int = 5) -> list[OrderbookLevel]:
        """Create synthetic ask levels."""
        from btc_predictor.config import TICK_SIZE
        levels = []
        for i in range(n_levels):
            levels.append(OrderbookLevel(
                price=mid + (i + 1) * TICK_SIZE,
                size=100.0,
            ))
        return levels

    def test_fill_returns_fill_object(self):
        asks = self._make_asks(0.55)
        fill = simulate_fill("UP", 10.0, asks, 0.55, ts_s=1000, condition_id="test")
        assert fill is not None
        assert fill.direction == "UP"
        assert fill.usdc_spent > 0
        assert fill.usdc_spent <= 10.0
        assert fill.shares > 0
        assert fill.taker_fee > 0

    def test_win_pnl_positive(self):
        asks = self._make_asks(0.50)
        fill = simulate_fill("UP", 50.0, asks, 0.50, ts_s=1000, condition_id="test")
        assert fill is not None
        fill = resolve_fill(fill, resolved_up=True)
        assert fill.pnl_net > 0   # should profit when UP wins at ~0.50

    def test_loss_pnl_negative(self):
        asks = self._make_asks(0.50)
        fill = simulate_fill("UP", 50.0, asks, 0.50, ts_s=1000, condition_id="test")
        assert fill is not None
        fill = resolve_fill(fill, resolved_up=False)
        assert fill.pnl_net < 0   # lose when DOWN resolves
        assert abs(fill.pnl_net) == pytest.approx(
            fill.usdc_spent + fill.taker_fee + fill.slippage_cost, rel=0.01
        )

    def test_down_bet_wins_when_down_resolves(self):
        asks = self._make_asks(0.45)
        fill = simulate_fill("DOWN", 50.0, asks, 0.45, ts_s=1000, condition_id="test")
        assert fill is not None
        fill = resolve_fill(fill, resolved_up=False)
        assert fill.pnl_net > 0

    def test_no_liquidity_returns_none(self):
        fill = simulate_fill("UP", 10.0, [], 0.50, ts_s=1000, condition_id="test")
        assert fill is None

    def test_win_fee_applied(self):
        """Win fee should reduce profit by 2%."""
        asks = [OrderbookLevel(price=0.5001, size=1000.0)]
        fill = simulate_fill("UP", 100.0, asks, 0.50, ts_s=1000, condition_id="test")
        fill = resolve_fill(fill, resolved_up=True)
        # gross_profit ≈ 100 / 0.5 - 100 = 100
        # win_fee ≈ 0.02 * 100 = 2
        assert fill.pnl_gross == pytest.approx(fill.shares - fill.usdc_spent, rel=0.01)
        assert fill.pnl_net < fill.pnl_gross   # fees reduce net

    def test_partial_fill(self):
        """Only fill up to available liquidity."""
        asks = [OrderbookLevel(price=0.55, size=5.0)]   # only $5 available
        fill = simulate_fill("UP", 100.0, asks, 0.55, ts_s=1000, condition_id="test")
        assert fill is not None
        assert fill.usdc_spent == pytest.approx(5.0, rel=0.01)


class TestBreakEven:
    def test_break_even_near_50_pct(self):
        """At market price 0.50, break-even prob should be ~0.51 (above 50% for fees)."""
        be = break_even_edge(0.50)
        assert 0.50 < be < 0.55

    def test_break_even_increases_with_price(self):
        """Higher market price → higher break-even requirement."""
        be_50 = break_even_edge(0.50)
        be_60 = break_even_edge(0.60)
        be_70 = break_even_edge(0.70)
        assert be_50 < be_60 < be_70

    def test_ev_positive_above_break_even(self):
        """If model prob > break_even, EV should be positive."""
        market_price = 0.50
        be = break_even_edge(market_price)
        ev = expected_value_per_dollar(be + 0.02, market_price)
        assert ev > 0

    def test_ev_negative_below_break_even(self):
        """If model prob < market price, EV is negative."""
        market_price = 0.50
        ev = expected_value_per_dollar(0.48, market_price)
        assert ev < 0


class TestKellySizing:
    def test_kelly_positive_with_edge(self):
        """Kelly bet should be positive when we have edge."""
        bet = fractional_kelly(
            p_model=0.60,
            market_price=0.50,
            bankroll=1000.0,
        )
        assert bet > 0
        assert bet <= 100.0  # capped at MAX_KELLY_BET_USDC

    def test_kelly_zero_without_edge(self):
        """Kelly bet is zero when p_model ≤ market_price (no edge)."""
        bet = fractional_kelly(
            p_model=0.48,
            market_price=0.50,
            bankroll=1000.0,
        )
        assert bet == 0.0

    def test_kelly_capped_at_max(self):
        """Never bet more than MAX_KELLY_BET_USDC."""
        bet = fractional_kelly(
            p_model=0.99,
            market_price=0.01,
            bankroll=1_000_000.0,
        )
        from btc_predictor.config import MAX_KELLY_BET_USDC
        assert bet <= MAX_KELLY_BET_USDC

    def test_kelly_scales_with_edge(self):
        """Larger edge → larger Kelly bet."""
        bet_small_edge = fractional_kelly(0.52, 0.50, 1000.0)
        bet_large_edge = fractional_kelly(0.65, 0.50, 1000.0)
        assert bet_large_edge > bet_small_edge
