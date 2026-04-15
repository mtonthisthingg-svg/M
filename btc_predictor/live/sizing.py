"""
Fractional Kelly criterion position sizing.

Kelly formula for binary bets:
  f* = (p * odds - (1-p)) / odds
  f* = p - (1-p)/odds

Where:
  p    = model's estimated P(win)
  odds = (1/market_price - 1) * (1 - win_fee)   (net odds after fees)

We apply:
  1. Fractional Kelly: f = f* * kelly_fraction  (dampens variance)
  2. Hard cap: max_bet_usdc
  3. Floor: never bet if f < 0

Reference: Kelly (1956), Thorp (1969).
"""

from __future__ import annotations

from btc_predictor.config import (
    KELLY_FRACTION,
    MAX_KELLY_BET_USDC,
    POLYMARKET_TAKER_FEE,
    POLYMARKET_WIN_FEE,
    SLIPPAGE_TICKS,
    TICK_SIZE,
)


def fractional_kelly(
    p_model: float,
    market_price: float,
    bankroll: float,
    kelly_fraction: float = KELLY_FRACTION,
    max_bet: float = MAX_KELLY_BET_USDC,
) -> float:
    """
    Compute bet size in USDC using fractional Kelly.

    Args:
        p_model:       Model's P(outcome wins), in (0, 1).
        market_price:  Market price of the outcome token, in (0, 1).
        bankroll:      Current total capital in USDC.
        kelly_fraction: Fraction of full Kelly to use (0.25 default).
        max_bet:       Hard cap on bet size.

    Returns:
        Bet size in USDC (>= 0).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if p_model <= 0 or p_model >= 1:
        return 0.0

    # Effective market price after entry slippage
    effective_price = market_price + SLIPPAGE_TICKS * TICK_SIZE
    effective_price = min(effective_price, 0.99)

    # Net odds per $ bet: win X/p - X, minus win fee
    net_win  = (1.0 / effective_price - 1.0) * (1.0 - POLYMARKET_WIN_FEE)

    # Taker fee reduces expected value regardless
    net_loss = 1.0 + POLYMARKET_TAKER_FEE    # lose bet + taker fee

    # Adjusted Kelly
    f_star = (p_model * net_win - (1.0 - p_model) * net_loss) / (net_win + net_loss)

    if f_star <= 0:
        return 0.0

    bet = bankroll * f_star * kelly_fraction
    return min(bet, max_bet)


def expected_value_per_dollar(
    p_model: float,
    market_price: float,
) -> float:
    """
    Expected value per $1 bet, after all fees.
    Positive = edge, negative = losing bet.
    """
    if market_price <= 0 or market_price >= 1:
        return -1.0

    effective_price = market_price + SLIPPAGE_TICKS * TICK_SIZE
    effective_price = min(effective_price, 0.99)

    net_win = (1.0 / effective_price - 1.0) * (1.0 - POLYMARKET_WIN_FEE)

    return p_model * net_win - (1.0 - p_model) - POLYMARKET_TAKER_FEE
