"""
Polymarket data feed — CLOB orderbook + Gamma market metadata.

Key responsibilities:
1. Discover active BTC 5-min markets via Gamma API
2. Stream CLOB orderbook updates via WebSocket
3. Track current market odds (UP token price = implied P(up))
4. Expose: market_mid, spread, depth, open_price, time_remaining

Resolution rules (confirmed):
  - Market slug: btc-updown-5m-{unix_ts}  (unix_ts divisible by 300)
  - Open price:  Chainlink BTC/USD at the first tick at or after window open
  - Resolution:  Chainlink BTC/USD at close epoch vs open price
  - UP wins if close_chainlink >= open_chainlink
  - Fees: 2% on winnings + taker fee (~0.5%)

CLOB WebSocket: wss://clob.polymarket.com
  Channels: {type: "market", assets_ids: [...]}  → price level updates
Gamma REST:  https://gamma-api.polymarket.com/markets?slug=btc-updown-5m-{ts}
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from btc_predictor.config import (
    MARKET_SLUG_PREFIX,
    MARKET_WINDOW_SECONDS,
    POLYMARKET_CLOB_REST,
    POLYMARKET_CLOB_WS,
    POLYMARKET_GAMMA_REST,
    TICK_SIZE,
)


@dataclass
class PolymarketMarket:
    """Metadata for a single BTC up/down 5-min market."""
    condition_id: str
    question_id: str
    slug: str
    open_ts: int            # UTC epoch seconds
    close_ts: int           # UTC epoch seconds
    up_token_id: str        # condition token for UP outcome
    down_token_id: str      # condition token for DOWN outcome
    # Live state (updated by CLOB feed)
    up_bid: float = 0.0     # best bid for UP token
    up_ask: float = 0.0     # best ask for UP token
    down_bid: float = 0.0
    down_ask: float = 0.0
    up_mid: float = 0.5     # implied P(up)
    # CLOB orderbook depth (top-5 each side)
    up_bids: List[tuple[float, float]] = field(default_factory=list)
    up_asks: List[tuple[float, float]] = field(default_factory=list)
    # Volume
    up_volume: float = 0.0
    down_volume: float = 0.0
    # Chainlink open price (locked at market open)
    chainlink_open_price: float = 0.0
    last_update_ts: float = 0.0

    @property
    def time_remaining_s(self) -> float:
        return max(0.0, self.close_ts - time.time())

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.open_ts <= now < self.close_ts

    @property
    def spread(self) -> float:
        if self.up_ask > 0 and self.up_bid > 0:
            return self.up_ask - self.up_bid
        return 1.0

    @property
    def mid(self) -> float:
        """Implied P(up) from the UP token mid-price."""
        if self.up_ask > 0 and self.up_bid > 0:
            return (self.up_ask + self.up_bid) / 2.0
        return 0.5


def _next_market_open_ts() -> int:
    """Return the next 5-min boundary epoch."""
    now = int(time.time())
    remainder = now % MARKET_WINDOW_SECONDS
    return now + (MARKET_WINDOW_SECONDS - remainder)


def _slug_for_ts(ts: int) -> str:
    return f"{MARKET_SLUG_PREFIX}-{ts}"


class PolymarketFeed:
    """
    Manages discovery and live orderbook streaming for Polymarket BTC markets.

    Usage:
        feed = PolymarketFeed()
        await feed.start()
        market = feed.current_market   # PolymarketMarket or None
        mid = feed.implied_prob_up     # float in [0,1]
    """

    def __init__(self) -> None:
        self._markets: Dict[str, PolymarketMarket] = {}   # slug → market
        self._current_slug: Optional[str] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []
        # Maps asset_id → market slug for WS routing
        self._asset_to_slug: Dict[str, str] = {}

    @property
    def current_market(self) -> Optional[PolymarketMarket]:
        if self._current_slug:
            return self._markets.get(self._current_slug)
        return None

    @property
    def implied_prob_up(self) -> float:
        m = self.current_market
        return m.mid if m else 0.5

    async def start(self) -> None:
        self._running = True
        self._tasks.append(asyncio.create_task(self._market_discovery_loop()))
        self._tasks.append(asyncio.create_task(self._clob_ws_loop()))

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------

    async def _market_discovery_loop(self) -> None:
        """Poll Gamma API for upcoming markets every 30 seconds."""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            while self._running:
                try:
                    await self._discover_markets(session)
                except Exception as e:
                    logger.warning(f"[PolymarketFeed] discovery error: {e}")
                await asyncio.sleep(30)

    async def _discover_markets(self, session: aiohttp.ClientSession) -> None:
        """Fetch current + next BTC 5-min markets from Gamma API."""
        now_ts = int(time.time())
        # Round down to current window
        current_open = now_ts - (now_ts % MARKET_WINDOW_SECONDS)

        for offset in [0, MARKET_WINDOW_SECONDS]:
            ts = current_open + offset
            slug = _slug_for_ts(ts)
            if slug in self._markets:
                continue
            market = await self._fetch_market(session, slug, ts)
            if market:
                self._markets[slug] = market
                self._asset_to_slug[market.up_token_id]   = slug
                self._asset_to_slug[market.down_token_id] = slug
                logger.info(
                    f"[PolymarketFeed] Discovered market {slug} "
                    f"open={market.open_ts} close={market.close_ts}"
                )

        # Update current slug
        active = [
            (slug, m) for slug, m in self._markets.items()
            if m.is_active
        ]
        if active:
            self._current_slug = active[0][0]

    async def _fetch_market(
        self,
        session: aiohttp.ClientSession,
        slug: str,
        ts: int,
    ) -> Optional[PolymarketMarket]:
        """Fetch market from Gamma API by slug."""
        url = f"{POLYMARKET_GAMMA_REST}/markets?slug={slug}"
        try:
            async with session.get(url) as r:
                if r.status != 200:
                    return None
                data = await r.json()
        except Exception as e:
            logger.debug(f"[PolymarketFeed] fetch {slug} failed: {e}")
            return None

        if not data:
            return None

        m = data[0] if isinstance(data, list) else data

        # Parse tokens
        tokens = m.get("tokens", m.get("clobTokenIds", []))
        if len(tokens) < 2:
            return None

        up_token_id   = tokens[0] if isinstance(tokens[0], str) else tokens[0].get("token_id", "")
        down_token_id = tokens[1] if isinstance(tokens[1], str) else tokens[1].get("token_id", "")

        return PolymarketMarket(
            condition_id=m.get("conditionId", ""),
            question_id=m.get("questionId", ""),
            slug=slug,
            open_ts=ts,
            close_ts=ts + MARKET_WINDOW_SECONDS,
            up_token_id=up_token_id,
            down_token_id=down_token_id,
        )

    # ------------------------------------------------------------------
    # CLOB WebSocket orderbook
    # ------------------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(20),
        reraise=True,
    )
    async def _clob_ws_loop(self) -> None:
        import websockets

        logger.info(f"[PolymarketFeed] CLOB WS connecting to {POLYMARKET_CLOB_WS}")
        async with websockets.connect(
            POLYMARKET_CLOB_WS,
            ping_interval=20,
            ping_timeout=30,
        ) as ws:
            # Subscribe to all known asset IDs
            await self._subscribe_all(ws)
            last_sub_refresh = time.time()

            while self._running:
                # Resubscribe when new markets are discovered
                if time.time() - last_sub_refresh > 60:
                    await self._subscribe_all(ws)
                    last_sub_refresh = time.time()

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                self._handle_clob_message(msg)

    async def _subscribe_all(self, ws) -> None:
        asset_ids = list(self._asset_to_slug.keys())
        if not asset_ids:
            return
        sub = json.dumps({
            "auth": {},
            "markets": [],
            "assets_ids": asset_ids,
            "type": "market",
        })
        await ws.send(sub)
        logger.debug(f"[PolymarketFeed] Subscribed to {len(asset_ids)} assets")

    def _handle_clob_message(self, msg: dict) -> None:
        """Parse CLOB WS price level message and update market state."""
        asset_id = msg.get("asset_id", "")
        slug = self._asset_to_slug.get(asset_id)
        if not slug or slug not in self._markets:
            return

        market = self._markets[slug]
        event_type = msg.get("event_type", "")
        price_levels = msg.get("price_levels", [])

        if not price_levels:
            return

        is_up = (asset_id == market.up_token_id)

        # price_levels: [{"price": "0.55", "size": "100", "side": "BUY"}, ...]
        bids = sorted(
            [(float(pl["price"]), float(pl["size"]))
             for pl in price_levels if pl.get("side") == "BUY"],
            reverse=True,
        )
        asks = sorted(
            [(float(pl["price"]), float(pl["size"]))
             for pl in price_levels if pl.get("side") == "SELL"],
        )

        if is_up:
            market.up_bids = bids[:5]
            market.up_asks = asks[:5]
            market.up_bid = bids[0][0] if bids else 0.0
            market.up_ask = asks[0][0] if asks else 1.0
        else:
            market.down_bids = bids[:5]
            market.down_asks = asks[:5]
            market.down_bid = bids[0][0] if bids else 0.0
            market.down_ask = asks[0][0] if asks else 1.0

        market.last_update_ts = time.time()

    # ------------------------------------------------------------------
    # Convenience: get CLOB orderbook snapshot for a market
    # ------------------------------------------------------------------

    async def get_book_snapshot(self, market: PolymarketMarket) -> dict:
        """Fetch full CLOB orderbook via REST (for fill simulation)."""
        url = f"{POLYMARKET_CLOB_REST}/book?token_id={market.up_token_id}"
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as r:
                return await r.json()

    async def get_market_trades(
        self, market: PolymarketMarket, limit: int = 100
    ) -> List[dict]:
        """Fetch recent trades for a market via CLOB REST."""
        url = (
            f"{POLYMARKET_CLOB_REST}/trades"
            f"?market={market.condition_id}&limit={limit}"
        )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as session:
            async with session.get(url) as r:
                data = await r.json()
                return data if isinstance(data, list) else data.get("data", [])
