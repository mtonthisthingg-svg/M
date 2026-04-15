"""
Bybit V5 WebSocket feed — BTC-USDT spot trades + L2 orderbook (depth 20).

Provides a third exchange reference for multi-venue features.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.config import BYBIT_SYMBOL, BYBIT_WS
from btc_predictor.data.bar_builder import BarBuilder, MultiExchangeBarStore, Quote, Trade


class BybitFeed:
    EXCHANGE = "bybit"

    def __init__(self, store: MultiExchangeBarStore) -> None:
        self._store = store
        self._builder = BarBuilder(self.EXCHANGE)
        self._ob_bids: dict[float, float] = {}
        self._ob_asks: dict[float, float] = {}
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

        sub = json.dumps({
            "op": "subscribe",
            "args": [
                f"publicTrade.{BYBIT_SYMBOL}",
                f"orderbook.20.{BYBIT_SYMBOL}",
            ],
        })

        logger.info(f"[BybitFeed] Connecting to {BYBIT_WS}")
        async with websockets.connect(BYBIT_WS, ping_interval=20, ping_timeout=30) as ws:
            await ws.send(sub)
            logger.info("[BybitFeed] Subscribed")

            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    await self._maybe_flush()
                    continue

                msg = json.loads(raw)
                topic = msg.get("topic", "")
                data  = msg.get("data", {})
                ts_ms = msg.get("ts", int(time.time() * 1000))

                if topic == f"publicTrade.{BYBIT_SYMBOL}":
                    for t in (data if isinstance(data, list) else [data]):
                        trade = Trade(
                            ts_ns=int(t.get("T", ts_ms)) * 1_000_000,
                            price=float(t["p"]),
                            qty=float(t["v"]),
                            # Bybit: S="Sell" means taker sold → is_buyer_maker=True
                            is_buyer_maker=(t.get("S", "") == "Sell"),
                            exchange=self.EXCHANGE,
                        )
                        self._builder.on_trade(trade)

                elif topic == f"orderbook.20.{BYBIT_SYMBOL}":
                    msg_type = msg.get("type", "delta")
                    if msg_type == "snapshot":
                        self._ob_bids = {float(p): float(q) for p, q in data.get("b", [])}
                        self._ob_asks = {float(p): float(q) for p, q in data.get("a", [])}
                    else:
                        for p, q in data.get("b", []):
                            price, qty = float(p), float(q)
                            if qty == 0:
                                self._ob_bids.pop(price, None)
                            else:
                                self._ob_bids[price] = qty
                        for p, q in data.get("a", []):
                            price, qty = float(p), float(q)
                            if qty == 0:
                                self._ob_asks.pop(price, None)
                            else:
                                self._ob_asks[price] = qty

                    bids = sorted(self._ob_bids.items(), reverse=True)[:20]
                    asks = sorted(self._ob_asks.items())[:20]
                    self._builder.on_quote(Quote(
                        ts_ns=ts_ms * 1_000_000,
                        bids=bids,
                        asks=asks,
                        exchange=self.EXCHANGE,
                    ))

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
