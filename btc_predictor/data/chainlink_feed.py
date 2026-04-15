"""
Polymarket Real-Time Data Socket (RTDS) — Chainlink BTC/USD feed.

WebSocket: wss://ws-live-data.polymarket.com
Topic:     crypto_prices_chainlink
Symbol:    btc/usd

Message format (observed):
  {
    "topic": "crypto_prices_chainlink",
    "type": "update",
    "timestamp": 1753314088421,         # ms UTC (message level)
    "payload": {
      "symbol": "btc/usd",
      "timestamp": 1753314088395,       # ms UTC (data level)
      "value": 67234.50
    }
  }

This is the RESOLUTION ORACLE for 5-minute BTC markets.
The reference (open) price is the first tick at or after each 300s boundary.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.config import (
    POLYMARKET_RTDS_WS,
    RESOLUTION_SYMBOL,
    STALE_DATA_THRESHOLD_S,
)

# Callback type: (ts_ms: int, price: float) -> None
PriceCallback = Callable[[int, float], None]


class ChainlinkFeed:
    """
    Subscribes to Polymarket RTDS and streams Chainlink BTC/USD prices.

    The feed calls `on_price(ts_ms, price)` for every tick.
    It also tracks the last tick timestamp so callers can detect staleness.
    """

    def __init__(
        self,
        on_price: Optional[PriceCallback] = None,
        symbol: str = RESOLUTION_SYMBOL,
    ) -> None:
        self._on_price = on_price
        self._symbol = symbol
        self._last_tick_ts: float = 0.0       # wall-clock time of last tick
        self._latest_price: float = 0.0
        self._latest_ts_ms: int = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def latest(self) -> tuple[int, float]:
        """Returns (ts_ms, price) of the most recent Chainlink tick."""
        return self._latest_ts_ms, self._latest_price

    @property
    def is_stale(self) -> bool:
        if self._last_tick_ts == 0.0:
            return True
        return (time.monotonic() - self._last_tick_ts) > STALE_DATA_THRESHOLD_S

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

        sub_msg = json.dumps({
            "action": "subscribe",
            "subscriptions": [
                {
                    "topic": "crypto_prices_chainlink",
                    "type": "*",
                    "filters": json.dumps({"symbol": self._symbol}),
                }
            ],
        })

        logger.info(f"[ChainlinkFeed] Connecting to {POLYMARKET_RTDS_WS}")
        async with websockets.connect(
            POLYMARKET_RTDS_WS,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:
            await ws.send(sub_msg)
            logger.info(f"[ChainlinkFeed] Subscribed to {self._symbol}")

            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    if not self._running:
                        break
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("topic") != "crypto_prices_chainlink":
                    continue
                if msg.get("type") != "update":
                    continue

                payload = msg.get("payload", {})
                sym = payload.get("symbol", "")
                if sym.lower() != self._symbol.lower():
                    continue

                price = float(payload.get("value", 0))
                ts_ms = int(payload.get("timestamp", msg.get("timestamp", 0)))

                if price <= 0:
                    continue

                self._latest_price = price
                self._latest_ts_ms = ts_ms
                self._last_tick_ts = time.monotonic()

                if self._on_price:
                    self._on_price(ts_ms, price)

                logger.debug(
                    f"[Chainlink] BTC/USD={price:.2f}  ts={ts_ms}"
                )
