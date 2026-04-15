"""
Fill simulation for Polymarket binary prediction markets.

Cost model:
  - Taker fee: 0.5% of order value (added to 15/5-min markets in 2025)
  - Win fee:   2% of winnings (Polymarket takes 2% of net profit)
  - Slippage:  we cross the spread + assumed price impact
  - Partial fills: if orderbook depth < desired size, fill what's available

Polymarket tokens trade between $0.01 and $0.99.
A bet of $X on UP at price p:
  - Cost:           X dollars (p per share, X/p shares)
  - If UP resolves: receive X/p * $1.00 = X/p
  - Net profit:     X/p - X = X*(1/p - 1)
  - Win fee:        0.02 * X*(1/p - 1)
  - Taker fee:      0.005 * X (paid at entry)
  - Net after fees: X*(1/p - 1) * (1 - 0.02) - 0.005*X
  - If DOWN:        lose X

Break-even edge:
  Let p = market price (implied prob).
  Model estimates true prob = q.
  Expected value per $ bet = q*(1/p - 1)*(0.98) + (1-q)*(-1) - 0.005
  Set EV > 0:
    q > (1 + 0.005) / (0.98/p - 0.98 + 1)
    ≈ p + fees + slippage  (approx for small fees)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from btc_predictor.config import (
    MAX_POSITION_USDC,
    POLYMARKET_TAKER_FEE,
    POLYMARKET_WIN_FEE,
    SLIPPAGE_TICKS,
    TICK_SIZE,
)


@dataclass
class Fill:
    """Result of a simulated fill."""
    ts_s: int
    condition_id: str
    direction: str             # "UP" or "DOWN"
    shares: float              # number of shares bought
    entry_price: float         # price paid per share (after slippage)
    usdc_spent: float          # = shares * entry_price
    taker_fee: float           # paid at entry
    slippage_cost: float       # estimated slippage vs mid
    resolved_up: Optional[bool] = None   # set at resolution
    pnl_gross: float = 0.0     # gross PnL before win fee
    pnl_net: float = 0.0       # net PnL after all fees


@dataclass
class OrderbookLevel:
    price: float
    size: float     # in USDC (Polymarket quotes in $ terms)


def simulate_fill(
    direction: str,
    usdc_budget: float,
    orderbook: List[OrderbookLevel],  # asks if buying UP, bids if selling
    mid_price: float,
    ts_s: int,
    condition_id: str,
) -> Optional[Fill]:
    """
    Simulate a market-order fill crossing the spread.

    Args:
        direction:    "UP" or "DOWN"
        usdc_budget:  Maximum $ to spend.
        orderbook:    Ask levels (best first) if buying UP.
        mid_price:    Current mid price (for slippage calculation).
        ts_s:         Timestamp of fill.
        condition_id: Market identifier.

    Returns:
        Fill object, or None if no liquidity.
    """
    usdc_budget = min(usdc_budget, MAX_POSITION_USDC)

    shares_filled  = 0.0
    usdc_filled    = 0.0
    weighted_price = 0.0

    for level in orderbook:
        if usdc_filled >= usdc_budget:
            break
        remaining_budget = usdc_budget - usdc_filled
        # Slippage: add SLIPPAGE_TICKS ticks to the ask price
        effective_price = level.price + SLIPPAGE_TICKS * TICK_SIZE
        effective_price = min(effective_price, 0.99)   # clamp to market bounds

        # How many shares can we buy at this level?
        level_usdc_avail = level.size  # size in USDC
        level_usdc_to_use = min(remaining_budget, level_usdc_avail)

        if level_usdc_to_use <= 0:
            continue

        shares_this_level = level_usdc_to_use / effective_price
        weighted_price += effective_price * level_usdc_to_use
        shares_filled  += shares_this_level
        usdc_filled    += level_usdc_to_use

    if shares_filled == 0 or usdc_filled == 0:
        return None

    avg_price     = weighted_price / usdc_filled
    taker_fee     = usdc_filled * POLYMARKET_TAKER_FEE
    slippage_cost = max(0.0, (avg_price - mid_price) * shares_filled)

    return Fill(
        ts_s=ts_s,
        condition_id=condition_id,
        direction=direction,
        shares=shares_filled,
        entry_price=avg_price,
        usdc_spent=usdc_filled,
        taker_fee=taker_fee,
        slippage_cost=slippage_cost,
    )


def resolve_fill(fill: Fill, resolved_up: bool) -> Fill:
    """
    Apply resolution outcome to a fill.

    UP token: pays $1.00 if UP resolves, else $0.00.
    DOWN token: pays $1.00 if DOWN resolves, else $0.00.
    """
    won = (fill.direction == "UP" and resolved_up) or \
          (fill.direction == "DOWN" and not resolved_up)

    if won:
        gross_proceeds = fill.shares * 1.0                    # each share pays $1
        gross_profit   = gross_proceeds - fill.usdc_spent
        win_fee        = max(0.0, gross_profit * POLYMARKET_WIN_FEE)
        pnl_gross      = gross_profit
        pnl_net        = gross_profit - win_fee - fill.taker_fee - fill.slippage_cost
    else:
        pnl_gross = -fill.usdc_spent
        pnl_net   = -fill.usdc_spent - fill.taker_fee - fill.slippage_cost

    fill.resolved_up = resolved_up
    fill.pnl_gross   = pnl_gross
    fill.pnl_net     = pnl_net
    return fill


def break_even_edge(market_price: float) -> float:
    """
    Minimum model probability to have positive EV given fees and slippage.

    EV = q*(1/p - 1)*(1 - win_fee) - (1-q) - taker_fee - slippage
    Solve for EV = 0:
      q = (1 + taker_fee) / ((1/p - 1)*(1 - win_fee) + 1)
    """
    p = market_price
    if p <= 0 or p >= 1:
        return 1.0
    slippage_cost = SLIPPAGE_TICKS * TICK_SIZE
    effective_price = p + slippage_cost
    effective_price = min(effective_price, 0.98)

    odds = (1.0 / effective_price - 1.0) * (1.0 - POLYMARKET_WIN_FEE)
    if odds <= 0:
        return 1.0

    be = (1.0 + POLYMARKET_TAKER_FEE) / (odds + 1.0)
    return float(be)
