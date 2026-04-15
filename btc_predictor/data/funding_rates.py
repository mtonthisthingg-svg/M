"""
Funding rate collector — Binance and Bybit perpetual funding rates.

Funding rates reflect the cost of leverage and are a proxy for
speculative positioning. Extreme funding (>0.1%/8h) can predict
mean-reversion; near-zero funding removes carry bias.

Polling cadence: every 10 seconds (rates update every 8h but the
estimated next rate refreshes every few seconds).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp
from loguru import logger


@dataclass
class FundingSnapshot:
    ts_s: int
    binance_funding_rate: float      # as decimal, e.g. 0.0001
    binance_next_funding_ts_ms: int
    bybit_funding_rate: float
    bybit_next_funding_ts_ms: int


class FundingRateCollector:
    """Polls Binance and Bybit for BTCUSDT perpetual funding rates."""

    BINANCE_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"
    BYBIT_URL   = "https://api.bybit.com/v5/market/funding/history?category=linear&symbol=BTCUSDT&limit=1"
    BYBIT_TICKER = "https://api.bybit.com/v5/market/tickers?category=linear&symbol=BTCUSDT"

    def __init__(self, poll_interval_s: int = 10) -> None:
        self._poll_interval = poll_interval_s
        self._latest: Optional[FundingSnapshot] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def latest(self) -> Optional[FundingSnapshot]:
        return self._latest

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

    async def _run(self) -> None:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=8)
        ) as session:
            while self._running:
                try:
                    snap = await self._fetch(session)
                    if snap:
                        self._latest = snap
                except Exception as e:
                    logger.warning(f"[FundingRates] fetch error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _fetch(self, session: aiohttp.ClientSession) -> Optional[FundingSnapshot]:
        bnb_rate = 0.0
        bnb_next = 0
        bybit_rate = 0.0
        bybit_next = 0

        try:
            async with session.get(self.BINANCE_URL) as r:
                d = await r.json()
                bnb_rate = float(d.get("lastFundingRate", 0))
                bnb_next = int(d.get("nextFundingTime", 0))
        except Exception as e:
            logger.debug(f"[FundingRates] Binance error: {e}")

        try:
            async with session.get(self.BYBIT_TICKER) as r:
                d = await r.json()
                items = d.get("result", {}).get("list", [])
                if items:
                    bybit_rate = float(items[0].get("fundingRate", 0))
                    bybit_next = int(items[0].get("nextFundingTime", 0))
        except Exception as e:
            logger.debug(f"[FundingRates] Bybit error: {e}")

        return FundingSnapshot(
            ts_s=int(time.time()),
            binance_funding_rate=bnb_rate,
            binance_next_funding_ts_ms=bnb_next,
            bybit_funding_rate=bybit_rate,
            bybit_next_funding_ts_ms=bybit_next,
        )
