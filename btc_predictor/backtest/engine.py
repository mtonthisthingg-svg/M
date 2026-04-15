"""
Event-driven backtester for Polymarket BTC 5-minute binary markets.

Replays historical data chronologically:
  - Chainlink price ticks (for feature computation and resolution)
  - Binance bar data (for features)
  - Polymarket CLOB orderbook snapshots (for fill simulation)

Timeline for each 5-minute window:
  T+0:    Market opens, chainlink open price locked
  T+0..T+240: Model runs every second, decides whether to enter
  T+120:  Entry cutoff (no new entries within 2 min of close)
  T+300:  Market closes, Chainlink price at T+300 is resolution price
          → fill.resolved_up = (cl_close >= cl_open)
          → PnL computed

Historical data format (parquet files in DATA_DIR/):
  bars_binance_YYYYMMDD.parquet      — 1-second Binance bars
  bars_coinbase_YYYYMMDD.parquet     — 1-second Coinbase bars
  chainlink_YYYYMMDD.parquet         — Chainlink tick data (ts_ms, price)
  polymarket_books_YYYYMMDD.parquet  — CLOB snapshots (ts_s, condition_id, ...)
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from btc_predictor.backtest.fills import (
    Fill,
    OrderbookLevel,
    break_even_edge,
    resolve_fill,
    simulate_fill,
)
from btc_predictor.backtest.report import BacktestReport
from btc_predictor.config import (
    DATA_DIR,
    ENTRY_CUTOFF_S,
    KELLY_FRACTION,
    MARKET_WINDOW_SECONDS,
    MAX_KELLY_BET_USDC,
    MIN_EDGE,
    SAFETY_MARGIN,
)
from btc_predictor.live.sizing import fractional_kelly


@dataclass
class HistoricalMarket:
    """One 5-minute market in the replay."""
    open_ts: int
    close_ts: int
    condition_id: str
    chainlink_open: float = 0.0
    chainlink_close: float = 0.0
    resolved_up: Optional[bool] = None


class BacktestEngine:
    """
    Event-driven backtester. Operates on pre-loaded pandas DataFrames.

    Usage:
        engine = BacktestEngine(model, start_date, end_date)
        report = engine.run()
    """

    def __init__(
        self,
        model,               # EnsemblePredictor
        start_date: str,     # "YYYY-MM-DD"
        end_date:   str,
        data_dir:   str = DATA_DIR,
        initial_bankroll: float = 1000.0,
        kelly_fraction:   float = KELLY_FRACTION,
        min_edge:         float = MIN_EDGE,
        entry_cutoff_s:   int   = ENTRY_CUTOFF_S,
        safety_margin:    float = SAFETY_MARGIN,
    ) -> None:
        self._model         = model
        self._start         = pd.Timestamp(start_date, tz="UTC")
        self._end           = pd.Timestamp(end_date,   tz="UTC")
        self._data_dir      = Path(data_dir)
        self._bankroll      = initial_bankroll
        self._kelly_frac    = kelly_fraction
        self._min_edge      = min_edge
        self._entry_cutoff  = entry_cutoff_s
        self._safety_margin = safety_margin

    def run(self) -> BacktestReport:
        """Execute the full backtest. Returns a BacktestReport."""
        logger.info(f"[Backtest] {self._start.date()} → {self._end.date()}")

        # Load data
        bars_df, cl_df, pm_df = self._load_data()
        if bars_df.empty or cl_df.empty:
            raise ValueError("No data loaded — run scripts/fetch_history.py first")

        fills:   List[Fill] = []
        markets: List[HistoricalMarket] = []

        # Build market windows from Chainlink data
        cl_df = cl_df.sort_values("ts_ms").reset_index(drop=True)
        cl_df["ts_s"] = (cl_df["ts_ms"] / 1000).astype(int)

        first_ts = int(cl_df["ts_s"].min())
        last_ts  = int(cl_df["ts_s"].max())
        window_starts = range(
            first_ts - (first_ts % MARKET_WINDOW_SECONDS),
            last_ts,
            MARKET_WINDOW_SECONDS,
        )

        for open_ts in window_starts:
            close_ts = open_ts + MARKET_WINDOW_SECONDS

            # Chainlink open price = first tick at or after open_ts
            cl_window = cl_df[cl_df["ts_s"].between(open_ts, close_ts)]
            if cl_window.empty:
                continue

            cl_open  = float(cl_window.iloc[0]["price"])
            cl_close_row = cl_df[cl_df["ts_s"] >= close_ts]
            if cl_close_row.empty:
                continue
            cl_close = float(cl_close_row.iloc[0]["price"])
            resolved_up = cl_close >= cl_open

            market = HistoricalMarket(
                open_ts=open_ts,
                close_ts=close_ts,
                condition_id=f"btc-updown-5m-{open_ts}",
                chainlink_open=cl_open,
                chainlink_close=cl_close,
                resolved_up=resolved_up,
            )
            markets.append(market)

            # Simulate trading within this window
            market_fills = self._simulate_market(
                market, bars_df, cl_df, pm_df
            )
            for f in market_fills:
                f = resolve_fill(f, resolved_up)
                fills.append(f)
                self._bankroll += f.pnl_net

        logger.info(f"[Backtest] {len(markets)} markets, {len(fills)} trades")
        return BacktestReport(fills=fills, markets=markets, initial_bankroll=1000.0)

    def _simulate_market(
        self,
        market: HistoricalMarket,
        bars_df: pd.DataFrame,
        cl_df: pd.DataFrame,
        pm_df: pd.DataFrame,
    ) -> List[Fill]:
        """Run the model for each second within a market window."""
        fills: List[Fill] = []
        entered = False   # one entry per market

        for ts_s in range(market.open_ts + 5, market.close_ts - self._entry_cutoff):
            # Build feature snapshot at ts_s using only data strictly before ts_s
            features, sequence = self._build_features_at(
                ts_s, market, bars_df, cl_df
            )
            if features is None:
                continue

            # Model probability
            p = self._model.predict(features, sequence)

            # Get current market mid from Polymarket snapshot
            pm_snap = self._get_pm_snapshot(ts_s, market.condition_id, pm_df)
            if pm_snap is None:
                continue

            market_mid = pm_snap["mid"]
            edge_up   = p - market_mid
            edge_down = (1.0 - p) - (1.0 - market_mid)  # edge on DOWN

            be = break_even_edge(market_mid)
            min_required_edge = (be - market_mid) + self._safety_margin

            # Decide direction
            direction = None
            edge = 0.0
            if edge_up > min_required_edge + self._min_edge:
                direction = "UP"
                edge = edge_up
                ob_levels = [
                    OrderbookLevel(p=r["ask_price"], size=r["ask_size"])
                    for _, r in pm_snap.get("asks", pd.DataFrame()).iterrows()
                ] if isinstance(pm_snap.get("asks"), pd.DataFrame) else []
            elif edge_down > min_required_edge + self._min_edge:
                direction = "DOWN"
                edge = edge_down
                ob_levels = [
                    OrderbookLevel(p=r["ask_price"], size=r["ask_size"])
                    for _, r in pm_snap.get("down_asks", pd.DataFrame()).iterrows()
                ] if isinstance(pm_snap.get("down_asks"), pd.DataFrame) else []

            if direction is None or entered:
                continue

            # Kelly sizing
            bet_usdc = fractional_kelly(
                p_model=p if direction == "UP" else (1 - p),
                market_price=market_mid if direction == "UP" else (1 - market_mid),
                bankroll=self._bankroll,
                kelly_fraction=self._kelly_frac,
                max_bet=MAX_KELLY_BET_USDC,
            )

            if bet_usdc < 1.0:
                continue

            # Fallback orderbook: use synthetic levels from mid
            if not ob_levels:
                ob_levels = _synthetic_levels(market_mid, bet_usdc)

            fill = simulate_fill(
                direction=direction,
                usdc_budget=bet_usdc,
                orderbook=ob_levels,
                mid_price=market_mid,
                ts_s=ts_s,
                condition_id=market.condition_id,
            )
            if fill:
                fills.append(fill)
                entered = True
                logger.debug(
                    f"[Backtest] {market.condition_id} {direction} "
                    f"p={p:.3f} m={market_mid:.3f} edge={edge:.3f} "
                    f"bet=${bet_usdc:.1f}"
                )

        return fills

    def _build_features_at(
        self,
        ts_s: int,
        market: HistoricalMarket,
        bars_df: pd.DataFrame,
        cl_df: pd.DataFrame,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        Build flat feature dict and sequence from historical data at ts_s.
        Uses only rows with ts_s strictly < ts_s.
        """
        from btc_predictor.features.returns import compute_return_features, compute_acceleration
        from btc_predictor.features.volatility import compute_volatility_features
        from btc_predictor.features.orderbook import compute_orderbook_features
        from btc_predictor.features.flow import VPINCalculator, compute_flow_features
        from btc_predictor.data.bar_builder import Bar

        # Filter Binance bars strictly before ts_s
        lookback = 200
        bar_rows = bars_df[
            (bars_df["ts_s"] >= ts_s - lookback) &
            (bars_df["ts_s"] < ts_s)
        ].sort_values("ts_s")

        if len(bar_rows) < 10:
            return None, None

        bars = [_row_to_bar(r) for _, r in bar_rows.iterrows()]

        # Latest Chainlink price strictly before ts_s
        cl_rows = cl_df[cl_df["ts_s"] < ts_s]
        cl_price = float(cl_rows.iloc[-1]["price"]) if not cl_rows.empty else 0.0
        binance_mid = bars[-1].midprice if bars else 0.0
        oracle_lead = binance_mid - cl_price if cl_price > 0 else 0.0
        oracle_lead_bps = oracle_lead / cl_price * 10_000 if cl_price > 0 else 0.0

        # Features
        feats: Dict = {}
        feats.update(compute_return_features(bars, ts_s))
        feats.update(compute_acceleration(bars, ts_s))
        feats.update(compute_volatility_features(bars, ts_s))
        latest_bar = bars[-1] if bars else None
        ob = compute_orderbook_features(latest_bar)
        feats.update({f"bnb_{k}": v for k, v in ob.items()})
        feats.update({f"cb_{k}":  v for k, v in ob.items()})  # placeholder
        feats["cross_obi"] = 0.0

        vpin = VPINCalculator()
        for b in bars:
            vpin.update(b.buy_volume, b.sell_volume)
        feats.update(compute_flow_features(bars, ts_s, vpin))

        feats["oracle_lead"]     = oracle_lead
        feats["oracle_lead_bps"] = oracle_lead_bps

        # Polymarket features
        time_remaining = market.close_ts - ts_s
        open_gap_bps = (
            (cl_price - market.chainlink_open) / market.chainlink_open * 10_000
            if market.chainlink_open > 0 and cl_price > 0 else 0.0
        )
        feats["pm_implied_up"]     = 0.5   # will be overridden by pm_snap
        feats["pm_spread"]         = 0.02
        feats["pm_time_remaining"] = float(time_remaining)
        feats["pm_time_frac"]      = 1.0 - time_remaining / MARKET_WINDOW_SECONDS
        feats["pm_open_gap_bps"]   = open_gap_bps
        feats["pm_book_depth_up"]  = 0.0
        feats["funding_binance"]   = 0.0
        feats["funding_bybit"]     = 0.0
        feats["cme_basis_ann"]     = 0.0

        # Sequence (60s of bars)
        from btc_predictor.config import SEQUENCE_LENGTH_S
        seq_bars = [b for b in bars if b.ts_s >= ts_s - SEQUENCE_LENGTH_S]
        sequence = None
        if len(seq_bars) >= SEQUENCE_LENGTH_S:
            keys = sorted(feats.keys())
            rows = []
            for b in seq_bars[-SEQUENCE_LENGTH_S:]:
                row_feats = {k: feats.get(k, 0.0) for k in keys}
                rows.append([row_feats[k] for k in keys])
            sequence = np.array(rows, dtype=np.float32)

        return feats, sequence

    def _get_pm_snapshot(
        self,
        ts_s: int,
        condition_id: str,
        pm_df: pd.DataFrame,
    ) -> Optional[dict]:
        """Get the Polymarket orderbook snapshot closest to ts_s."""
        if pm_df.empty:
            # Synthetic: use 50/50 mid
            return {"mid": 0.50, "spread": 0.02, "asks": []}

        rows = pm_df[
            (pm_df["condition_id"] == condition_id) &
            (pm_df["ts_s"] <= ts_s)
        ]
        if rows.empty:
            return None

        row = rows.iloc[-1]
        return {
            "mid":    float(row.get("up_mid", 0.5)),
            "spread": float(row.get("spread", 0.02)),
            "asks":   [],
        }

    def _load_data(self):
        """Load parquet files for the date range."""
        bars_parts = []
        cl_parts   = []
        pm_parts   = []

        current = self._start
        while current <= self._end:
            date_str = current.strftime("%Y%m%d")
            bars_f = self._data_dir / f"bars_binance_{date_str}.parquet"
            cl_f   = self._data_dir / f"chainlink_{date_str}.parquet"
            pm_f   = self._data_dir / f"polymarket_books_{date_str}.parquet"

            if bars_f.exists():
                bars_parts.append(pd.read_parquet(bars_f))
            if cl_f.exists():
                cl_parts.append(pd.read_parquet(cl_f))
            if pm_f.exists():
                pm_parts.append(pd.read_parquet(pm_f))

            current += pd.Timedelta(days=1)

        bars_df = pd.concat(bars_parts, ignore_index=True) if bars_parts else pd.DataFrame()
        cl_df   = pd.concat(cl_parts,   ignore_index=True) if cl_parts   else pd.DataFrame()
        pm_df   = pd.concat(pm_parts,   ignore_index=True) if pm_parts   else pd.DataFrame()

        logger.info(f"[Backtest] Loaded {len(bars_df)} bars, {len(cl_df)} cl ticks, {len(pm_df)} pm snaps")
        return bars_df, cl_df, pm_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_bar(row) -> "Bar":
    from btc_predictor.data.bar_builder import Bar
    import numpy as np

    b = Bar(
        ts_s=int(row["ts_s"]),
        exchange=str(row.get("exchange", "binance")),
        open=float(row.get("open", 0)),
        high=float(row.get("high", 0)),
        low=float(row.get("low", 0)),
        close=float(row.get("close", 0)),
        volume=float(row.get("volume", 0)),
        buy_volume=float(row.get("buy_volume", 0)),
        sell_volume=float(row.get("sell_volume", 0)),
        trade_count=int(row.get("trade_count", 0)),
    )
    for attr in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"):
        val = row.get(attr)
        if val is not None and hasattr(val, "__len__"):
            setattr(b, attr, np.array(val, dtype=np.float64))
    b.midprice   = float(row.get("midprice", 0))
    b.spread     = float(row.get("spread", 0))
    b.microprice = float(row.get("microprice", 0))
    return b


def _synthetic_levels(mid: float, budget: float) -> List[OrderbookLevel]:
    """Generate synthetic ask levels around mid for backtest fills."""
    from btc_predictor.config import TICK_SIZE
    levels = []
    for i in range(5):
        price = mid + (i + 1) * TICK_SIZE
        price = min(price, 0.99)
        levels.append(OrderbookLevel(price=price, size=budget / 5))
    return levels
