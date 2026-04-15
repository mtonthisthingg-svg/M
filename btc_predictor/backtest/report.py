"""
Backtest performance reporting.

Reports:
  - Total PnL, annualized return
  - Sharpe ratio (daily PnL series)
  - Maximum drawdown
  - Hit rate (fraction of winning trades)
  - Edge vs market (actual win rate vs implied probability)
  - Brier score of model vs outcomes
  - Per-market breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestReport:
    fills: List["Fill"]
    markets: List["HistoricalMarket"]
    initial_bankroll: float = 1000.0

    def summary(self) -> dict:
        if not self.fills:
            return {"error": "No fills"}

        pnl_list = [f.pnl_net for f in self.fills]
        cumulative = np.cumsum(pnl_list)
        total_pnl  = cumulative[-1]

        # Daily PnL for Sharpe
        fill_df = pd.DataFrame([{
            "ts_s":    f.ts_s,
            "pnl_net": f.pnl_net,
        } for f in self.fills])
        fill_df["date"] = pd.to_datetime(fill_df["ts_s"], unit="s").dt.date
        daily_pnl = fill_df.groupby("date")["pnl_net"].sum()

        sharpe = _sharpe(daily_pnl.values)
        max_dd = _max_drawdown(cumulative)
        hit_rate = np.mean([f.pnl_net > 0 for f in self.fills])

        # Edge vs market
        edge_list = []
        for f in self.fills:
            if f.resolved_up is not None:
                direction_win = (
                    (f.direction == "UP"   and f.resolved_up) or
                    (f.direction == "DOWN" and not f.resolved_up)
                )
                edge_list.append(1 if direction_win else 0)
        edge_vs_market = np.mean(edge_list) - 0.5 if edge_list else 0.0

        n_days = len(daily_pnl)
        ann_return = (total_pnl / self.initial_bankroll) / max(n_days, 1) * 365

        return {
            "total_pnl":         round(total_pnl, 2),
            "total_pnl_pct":     round(total_pnl / self.initial_bankroll * 100, 2),
            "ann_return_pct":    round(ann_return * 100, 2),
            "sharpe":            round(sharpe, 3),
            "max_drawdown_pct":  round(max_dd * 100, 2),
            "hit_rate":          round(float(hit_rate), 4),
            "edge_vs_50pct":     round(float(edge_vs_market), 4),
            "n_trades":          len(self.fills),
            "n_markets":         len(self.markets),
            "n_days":            n_days,
            "avg_bet_usdc":      round(float(np.mean([f.usdc_spent for f in self.fills])), 2),
            "avg_pnl_per_trade": round(float(np.mean(pnl_list)), 3),
            "total_fees":        round(float(sum(f.taker_fee for f in self.fills)), 2),
        }

    def daily_pnl_series(self) -> pd.Series:
        if not self.fills:
            return pd.Series(dtype=float)
        df = pd.DataFrame([{"ts_s": f.ts_s, "pnl_net": f.pnl_net} for f in self.fills])
        df["date"] = pd.to_datetime(df["ts_s"], unit="s").dt.date
        return df.groupby("date")["pnl_net"].sum()

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_s":         f.ts_s,
            "condition_id": f.condition_id,
            "direction":    f.direction,
            "entry_price":  f.entry_price,
            "shares":       f.shares,
            "usdc_spent":   f.usdc_spent,
            "taker_fee":    f.taker_fee,
            "slippage":     f.slippage_cost,
            "resolved_up":  f.resolved_up,
            "pnl_gross":    f.pnl_gross,
            "pnl_net":      f.pnl_net,
        } for f in self.fills])

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 55)
        print("  BTC Polymarket Backtest Report")
        print("=" * 55)
        print(f"  Period:           {s['n_days']} days")
        print(f"  Markets seen:     {s['n_markets']}")
        print(f"  Trades:           {s['n_trades']}")
        print(f"  Avg bet:          ${s['avg_bet_usdc']:.2f} USDC")
        print(f"  Total PnL:        ${s['total_pnl']:.2f} ({s['total_pnl_pct']:.1f}%)")
        print(f"  Ann. return:      {s['ann_return_pct']:.1f}%")
        print(f"  Sharpe ratio:     {s['sharpe']:.3f}")
        print(f"  Max drawdown:     {s['max_drawdown_pct']:.1f}%")
        print(f"  Hit rate:         {s['hit_rate']:.1%}")
        print(f"  Edge vs 50%:      {s['edge_vs_50pct']:.1%}")
        print(f"  Avg PnL/trade:    ${s['avg_pnl_per_trade']:.3f}")
        print(f"  Total fees paid:  ${s['total_fees']:.2f}")
        print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _sharpe(daily_pnl: np.ndarray, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio from daily PnL series."""
    if len(daily_pnl) < 2:
        return 0.0
    excess = daily_pnl - risk_free / 365
    std = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * np.sqrt(365))


def _max_drawdown(cumulative_pnl: np.ndarray) -> float:
    """Maximum drawdown as fraction of peak equity."""
    if len(cumulative_pnl) == 0:
        return 0.0
    peak = cumulative_pnl[0]
    max_dd = 0.0
    for val in cumulative_pnl:
        if val > peak:
            peak = val
        dd = (peak - val) / (peak + 1e-9)
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)
