"""
Binance BTC-USDT WebSocket feed.

Streams:
  - aggTrade   → individual trades (price, qty, side)
  - depth20@100ms  → top-20 L2 orderbook snapshot

Binance spot leads the Chainlink oracle, so the spread
  oracle_lead = binance_mid − chainlink_price
is the primary predictive signal.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.config import BINANCE_SYMBOL, BINANCE_WS
from btc_predictor.data.bar_builder import BarBuilder, MultiExchangeBarStore, Quote, Trade

# Combined stream URL: two streams multiplexed
_STREAMS = f"{BINANCE_SYMBOL.lower()}@aggTrade/{BINANCE_SYMBOL.lower()}@depth20@100ms"


class BinanceFeed:
    """
    Async WebSocket client for Binance BTC-USDT trades + L2 orderbook.

    Writes Bars into a MultiExchangeBarStore every second.
    """

    EXCHANGE = "binance"

    def __init__(self, store: MultiExchangeBarStore) -> None:
        self._store = store
        self._builder = BarBuilder(self.EXCHANGE)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_flush_s: int = 0

    async def start(self) -> None:
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(20),
        reraise=True,
    )
    async def _run(self) -> None:
        import websockets

        url = f"{BINANCE_WS}/{_STREAMS}"
        logger.info(f"[BinanceFeed] Connecting to {url}")

        async with websockets.connect(url, ping_interval=20, ping_timeout=30) as ws:
            logger.info("[BinanceFeed] Connected")
            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    await self._maybe_flush()
                    continue

                msg = json.loads(raw)
                # Binance combined stream wraps with {"stream": ..., "data": ...}
                data = msg.get("data", msg)
                event = data.get("e", "")

                if event == "aggTrade":
                    trade = Trade(
                        ts_ns=data["T"] * 1_000_000,  # ms → ns
                        price=float(data["p"]),
                        qty=float(data["q"]),
                        is_buyer_maker=bool(data["m"]),
                        exchange=self.EXCHANGE,
                    )
                    self._builder.on_trade(trade)

                elif event == "depthUpdate":
                    # depth20 snapshot: bids/asks are full replacement
                    bids = [(float(p), float(q)) for p, q in data.get("b", [])]
                    asks = [(float(p), float(q)) for p, q in data.get("a", [])]
                    # Sort: bids descending, asks ascending
                    bids.sort(key=lambda x: -x[0])
                    asks.sort(key=lambda x: x[0])
                    quote = Quote(
                        ts_ns=data.get("T", data.get("E", 0)) * 1_000_000,
                        bids=bids,
                        asks=asks,
                        exchange=self.EXCHANGE,
                    )
                    self._builder.on_quote(quote)

                await self._maybe_flush()

    async def _maybe_flush(self) -> None:
        now_s = int(time.time())
        if now_s > self._last_flush_s:
            bars = self._builder.flush(now_s)
            for bar in bars:
                await self._store.add_bar(bar)
            if bars:
                self._last_flush_s = bars[-1].ts_s
            elif self._last_flush_s == 0:
                self._last_flush_s = now_s - 1
