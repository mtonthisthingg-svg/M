"""
CME BTC futures basis collector.

Uses Binance CME-linked futures (BTCUSDT_YYMMDD) as a proxy for
institutional positioning / basis spread. The basis =
  (futures_price - spot_price) / spot_price * annualized

A positive basis indicates contango; negative = backwardation.
This feeds into the feature pipeline as a slow-moving signal.

Data source: Binance delivery futures API (no auth required).
CME actual prices require a subscription; this is a free proxy.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import aiohttp
from loguru import logger


@dataclass
class CMEBasisSnapshot:
    ts_s: int
    near_contract: str        # e.g. "BTCUSDT_250926"
    near_futures_px: float
    spot_px: float
    basis_raw: float          # futures - spot
    basis_annualized: float   # (basis_raw / spot_px) * (365 / days_to_expiry)
    days_to_expiry: int


class CMEBasisCollector:
    """
    Polls Binance delivery futures for the front-month BTC contract
    and computes the basis vs Binance spot.
    """

    FUTURES_EXCHANGE_INFO = "https://dapi.binance.com/dapi/v1/exchangeInfo"
    FUTURES_TICKER        = "https://dapi.binance.com/dapi/v1/ticker/price?symbol={symbol}"
    SPOT_TICKER           = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"

    def __init__(self, poll_interval_s: int = 30) -> None:
        self._poll_interval = poll_interval_s
        self._latest: Optional[CMEBasisSnapshot] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._front_contract: Optional[str] = None

    @property
    def latest(self) -> Optional[CMEBasisSnapshot]:
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
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            # Discover front-month contract once
            await self._discover_contract(session)
            while self._running:
                try:
                    snap = await self._fetch(session)
                    if snap:
                        self._latest = snap
                except Exception as e:
                    logger.warning(f"[CMEBasis] fetch error: {e}")
                await asyncio.sleep(self._poll_interval)

    async def _discover_contract(self, session: aiohttp.ClientSession) -> None:
        try:
            async with session.get(self.FUTURES_EXCHANGE_INFO) as r:
                info = await r.json()
            symbols = info.get("symbols", [])
            btc_deliveries = [
                s for s in symbols
                if s.get("baseAsset") == "BTC"
                and s.get("contractType") == "CURRENT_QUARTER"
            ]
            if btc_deliveries:
                self._front_contract = btc_deliveries[0]["symbol"]
                logger.info(f"[CMEBasis] Front contract: {self._front_contract}")
        except Exception as e:
            logger.warning(f"[CMEBasis] contract discovery failed: {e}")

    async def _fetch(self, session: aiohttp.ClientSession) -> Optional[CMEBasisSnapshot]:
        if not self._front_contract:
            await self._discover_contract(session)
            if not self._front_contract:
                return None

        try:
            url_f = self.FUTURES_TICKER.format(symbol=self._front_contract)
            async with session.get(url_f) as r:
                fd = await r.json()
            futures_px = float(fd[0]["price"] if isinstance(fd, list) else fd["price"])

            async with session.get(self.SPOT_TICKER) as r:
                sd = await r.json()
            spot_px = float(sd["price"])

            # Parse expiry from contract name e.g. BTCUSD_250926 → 2025-09-26
            days = self._days_to_expiry(self._front_contract)
            basis_raw = futures_px - spot_px
            basis_ann = (basis_raw / spot_px) * (365.0 / max(days, 1)) if spot_px else 0.0

            return CMEBasisSnapshot(
                ts_s=int(time.time()),
                near_contract=self._front_contract,
                near_futures_px=futures_px,
                spot_px=spot_px,
                basis_raw=basis_raw,
                basis_annualized=basis_ann,
                days_to_expiry=days,
            )
        except Exception as e:
            logger.debug(f"[CMEBasis] price fetch error: {e}")
            return None

    @staticmethod
    def _days_to_expiry(symbol: str) -> int:
        """Extract days to expiry from Binance delivery contract name."""
        try:
            # Format: BTCUSD_YYMMDD or BTCUSDT_YYMMDD
            suffix = symbol.split("_")[-1]
            if len(suffix) == 6:
                from datetime import datetime, timezone
                yy, mm, dd = int(suffix[:2]), int(suffix[2:4]), int(suffix[4:])
                year = 2000 + yy
                expiry = datetime(year, mm, dd, tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                return max(0, (expiry - now).days)
        except Exception:
            pass
        return 90   # fallback
