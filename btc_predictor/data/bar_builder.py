"""
1-second bar builder.

Consumes raw trade and quote events from exchange feeds and produces
aligned OHLCV + orderbook snapshot bars at BAR_SECONDS resolution.
All timestamps are UTC Unix seconds (integer).
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from btc_predictor.config import BAR_SECONDS


@dataclass
class Trade:
    ts_ns: int          # nanoseconds UTC
    price: float
    qty: float
    is_buyer_maker: bool   # True = sell-side aggressor (taker sold)
    exchange: str


@dataclass
class Quote:
    ts_ns: int
    bids: List[tuple[float, float]]   # [(price, qty), ...]  best-first
    asks: List[tuple[float, float]]
    exchange: str


@dataclass
class Bar:
    ts_s: int           # bar open epoch (integer UTC seconds)
    exchange: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    buy_volume: float   # taker-buy volume
    sell_volume: float  # taker-sell volume
    trade_count: int
    # Snapshot of L2 at bar close (top-20 each side)
    bid_prices: np.ndarray = field(default_factory=lambda: np.zeros(20))
    bid_sizes:  np.ndarray = field(default_factory=lambda: np.zeros(20))
    ask_prices: np.ndarray = field(default_factory=lambda: np.zeros(20))
    ask_sizes:  np.ndarray = field(default_factory=lambda: np.zeros(20))
    # Derived
    midprice: float = 0.0
    spread: float = 0.0
    microprice: float = 0.0


class BarBuilder:
    """
    Aggregates Trade and Quote events into fixed 1-second bars.

    Usage:
        builder = BarBuilder("binance")
        builder.on_trade(trade)
        builder.on_quote(quote)
        bars = builder.flush(now_s)   # returns completed bars
    """

    def __init__(self, exchange: str, bar_seconds: int = BAR_SECONDS) -> None:
        self.exchange = exchange
        self.bar_seconds = bar_seconds
        self._trades: List[Trade] = []
        self._latest_quote: Optional[Quote] = None
        self._last_close: float = 0.0

    def on_trade(self, trade: Trade) -> None:
        self._trades.append(trade)

    def on_quote(self, quote: Quote) -> None:
        self._latest_quote = quote

    def flush(self, up_to_s: int) -> List[Bar]:
        """Return all completed bars with ts_s < up_to_s."""
        if not self._trades and self._last_close == 0.0:
            return []

        completed_bars: List[Bar] = []

        # Group trades by 1-second bucket
        buckets: Dict[int, List[Trade]] = defaultdict(list)
        for t in self._trades:
            bucket = int(t.ts_ns // 1_000_000_000)
            if bucket < up_to_s:
                buckets[bucket].append(t)

        # Remove all trades that fall into completed buckets
        self._trades = [t for t in self._trades
                        if int(t.ts_ns // 1_000_000_000) >= up_to_s]

        if not buckets:
            return []

        for ts_s in sorted(buckets.keys()):
            trades = buckets[ts_s]
            prices = [t.price for t in trades]
            open_  = prices[0]
            high   = max(prices)
            low    = min(prices)
            close  = prices[-1]
            volume      = sum(t.qty for t in trades)
            buy_vol     = sum(t.qty for t in trades if not t.is_buyer_maker)
            sell_vol    = sum(t.qty for t in trades if t.is_buyer_maker)

            bar = Bar(
                ts_s=ts_s,
                exchange=self.exchange,
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
                buy_volume=buy_vol,
                sell_volume=sell_vol,
                trade_count=len(trades),
            )

            # Attach latest L2 snapshot
            if self._latest_quote is not None:
                q = self._latest_quote
                n = 20
                bp = np.array([p for p, _ in q.bids[:n]], dtype=np.float64)
                bs = np.array([s for _, s in q.bids[:n]], dtype=np.float64)
                ap = np.array([p for p, _ in q.asks[:n]], dtype=np.float64)
                as_ = np.array([s for _, s in q.asks[:n]], dtype=np.float64)

                pad = lambda a, n: np.pad(a, (0, max(0, n - len(a))))
                bar.bid_prices = pad(bp, n)
                bar.bid_sizes  = pad(bs, n)
                bar.ask_prices = pad(ap, n)
                bar.ask_sizes  = pad(as_, n)

                if len(bp) and len(ap):
                    bar.midprice   = (bp[0] + ap[0]) / 2.0
                    bar.spread     = ap[0] - bp[0]
                    # Microprice: size-weighted midprice
                    bid_sz = bs[0] if len(bs) else 0.0
                    ask_sz = as_[0] if len(as_) else 0.0
                    denom = bid_sz + ask_sz
                    if denom > 0:
                        bar.microprice = (ap[0] * bid_sz + bp[0] * ask_sz) / denom
                    else:
                        bar.microprice = bar.midprice

            self._last_close = close
            completed_bars.append(bar)

        return completed_bars


class MultiExchangeBarStore:
    """
    Thread-safe (asyncio) store that holds aligned 1-second bars
    from multiple exchanges and the Chainlink oracle feed.

    Consumers call get_aligned_window(ts_s, lookback_s) to get a
    DataFrame-ready dict of the last `lookback_s` bars.
    """

    def __init__(self, max_history_s: int = 600) -> None:
        self._max_history = max_history_s
        # exchange -> deque of Bars (newest last)
        self._bars: Dict[str, List[Bar]] = defaultdict(list)
        # chainlink price history: list of (ts_s, price)
        self._chainlink: List[tuple[int, float]] = []
        self._lock = asyncio.Lock()

    async def add_bar(self, bar: Bar) -> None:
        async with self._lock:
            store = self._bars[bar.exchange]
            store.append(bar)
            # Trim to max history
            cutoff = bar.ts_s - self._max_history
            while store and store[0].ts_s < cutoff:
                store.pop(0)

    async def add_chainlink(self, ts_s: int, price: float) -> None:
        async with self._lock:
            self._chainlink.append((ts_s, price))
            cutoff = ts_s - self._max_history
            while self._chainlink and self._chainlink[0][0] < cutoff:
                self._chainlink.pop(0)

    async def latest_chainlink(self) -> Optional[tuple[int, float]]:
        async with self._lock:
            if self._chainlink:
                return self._chainlink[-1]
            return None

    async def latest_bar(self, exchange: str) -> Optional[Bar]:
        async with self._lock:
            bars = self._bars.get(exchange, [])
            return bars[-1] if bars else None

    async def get_window(
        self, exchange: str, end_ts_s: int, lookback_s: int
    ) -> List[Bar]:
        """Return up to lookback_s bars ending at end_ts_s (exclusive)."""
        async with self._lock:
            bars = self._bars.get(exchange, [])
            start = end_ts_s - lookback_s
            return [b for b in bars if start <= b.ts_s < end_ts_s]

    async def get_chainlink_window(
        self, end_ts_s: int, lookback_s: int
    ) -> List[tuple[int, float]]:
        async with self._lock:
            start = end_ts_s - lookback_s
            return [(ts, p) for ts, p in self._chainlink
                    if start <= ts < end_ts_s]
