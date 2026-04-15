"""
Coinbase Advanced Trade WebSocket feed — BTC-USD trades + L2 orderbook.

Uses the Coinbase Advanced Trade WS API (v3 format).
Provides a second exchange reference for cross-venue OBI and flow features.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.config import COINBASE_SYMBOL, COINBASE_WS
from btc_predictor.data.bar_builder import BarBuilder, MultiExchangeBarStore, Quote, Trade


class CoinbaseFeed:
    """
    Async WebSocket feed for Coinbase BTC-USD.
    Populates the MultiExchangeBarStore with 1-second bars.
    """

    EXCHANGE = "coinbase"

    def __init__(self, store: MultiExchangeBarStore) -> None:
        self._store = store
        self._builder = BarBuilder(self.EXCHANGE)
        self._orderbook_bids: dict[float, float] = {}
        self._orderbook_asks: dict[float, float] = {}
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

        sub = {
            "type": "subscribe",
            "product_ids": [COINBASE_SYMBOL],
            "channel": "market_trades",
        }
        sub_l2 = {
            "type": "subscribe",
            "product_ids": [COINBASE_SYMBOL],
            "channel": "level2",
        }

        logger.info(f"[CoinbaseFeed] Connecting to {COINBASE_WS}")
        async with websockets.connect(COINBASE_WS, ping_interval=20, ping_timeout=30) as ws:
            await ws.send(json.dumps(sub))
            await ws.send(json.dumps(sub_l2))
            logger.info("[CoinbaseFeed] Subscribed")

            while self._running:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    await self._maybe_flush()
                    continue

                msg = json.loads(raw)
                channel = msg.get("channel", "")
                events  = msg.get("events", [])

                for event in events:
                    if channel == "market_trades":
                        for t in event.get("trades", []):
                            trade = Trade(
                                ts_ns=_parse_ts_ns(t.get("time", "")),
                                price=float(t["price"]),
                                qty=float(t["size"]),
                                # side="BUY" means taker bought → buyer_maker=False
                                is_buyer_maker=(t.get("side", "") == "SELL"),
                                exchange=self.EXCHANGE,
                            )
                            self._builder.on_trade(trade)

                    elif channel == "level2":
                        etype = event.get("type", "")
                        updates = event.get("updates", [])
                        for u in updates:
                            side  = u["side"]
                            price = float(u["price_level"])
                            size  = float(u["new_quantity"])
                            if side == "bid":
                                if size == 0:
                                    self._orderbook_bids.pop(price, None)
                                else:
                                    self._orderbook_bids[price] = size
                            else:
                                if size == 0:
                                    self._orderbook_asks.pop(price, None)
                                else:
                                    self._orderbook_asks[price] = size

                        self._emit_quote(msg.get("timestamp", ""))

                await self._maybe_flush()

    def _emit_quote(self, ts_str: str) -> None:
        bids = sorted(self._orderbook_bids.items(), reverse=True)[:20]
        asks = sorted(self._orderbook_asks.items())[:20]
        quote = Quote(
            ts_ns=_parse_ts_ns(ts_str),
            bids=bids,
            asks=asks,
            exchange=self.EXCHANGE,
        )
        self._builder.on_quote(quote)

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


def _parse_ts_ns(ts: str) -> int:
    """Parse ISO-8601 or RFC-3339 timestamp string to nanoseconds."""
    if not ts:
        return int(time.time() * 1e9)
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(ts.rstrip("Z")).replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1e9)
    except Exception:
        return int(time.time() * 1e9)
