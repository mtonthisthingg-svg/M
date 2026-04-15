#!/usr/bin/env python3
"""
Inspect live Polymarket BTC 5-minute market structure.

This script connects to both the Polymarket RTDS (Chainlink feed) and the
Gamma API to show the current market's resolution rules, oracle source,
and live orderbook state.

Run BEFORE building features to confirm the prediction target is correct.

Usage:
    python -m btc_predictor.scripts.inspect_market
    python -m btc_predictor.scripts.inspect_market --watch   # continuous

Key things to confirm:
  1. Resolution oracle: Chainlink BTC/USD (not raw Binance)
  2. Open price locking: first Chainlink tick at the 300s boundary
  3. Current implied probability matches CLOB mid
  4. Fee structure
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

# Allow running as script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from btc_predictor.config import (
    MARKET_WINDOW_SECONDS,
    POLYMARKET_CLOB_REST,
    POLYMARKET_GAMMA_REST,
    POLYMARKET_RTDS_WS,
    MARKET_SLUG_PREFIX,
)


async def inspect_live(watch: bool = False) -> None:
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        await _show_current_market(session)
        if watch:
            print("\n--- Watching live Chainlink feed (Ctrl-C to stop) ---\n")
            await _watch_chainlink()


async def _show_current_market(session: aiohttp.ClientSession) -> None:
    now = int(time.time())
    current_open = now - (now % MARKET_WINDOW_SECONDS)
    next_open    = current_open + MARKET_WINDOW_SECONDS

    print("=" * 60)
    print("  POLYMARKET BTC 5-MIN MARKET INSPECTOR")
    print("=" * 60)
    print(f"  Current time:   {_fmt_ts(now)}")
    print(f"  Window open:    {_fmt_ts(current_open)}")
    print(f"  Window close:   {_fmt_ts(next_open)}")
    print(f"  Time remaining: {next_open - now}s")
    print()

    # Fetch current market from Gamma API
    slug = f"{MARKET_SLUG_PREFIX}-{current_open}"
    print(f"  Fetching market: {slug}")

    try:
        url = f"{POLYMARKET_GAMMA_REST}/markets?slug={slug}"
        async with session.get(url) as r:
            if r.status != 200:
                print(f"  ERROR: Gamma API returned {r.status}")
                print(f"  (Market may not exist yet — trying next window slug)")
                slug = f"{MARKET_SLUG_PREFIX}-{next_open}"
                url = f"{POLYMARKET_GAMMA_REST}/markets?slug={slug}"
                async with session.get(url) as r2:
                    if r2.status != 200:
                        print(f"  ERROR: {r2.status} for {slug}")
                        return
                    data = await r2.json()
            else:
                data = await r.json()
    except Exception as e:
        print(f"  ERROR fetching market: {e}")
        return

    if not data:
        print("  No market data returned.")
        return

    m = data[0] if isinstance(data, list) else data

    print()
    print("  MARKET DETAILS")
    print("  " + "-" * 50)
    print(f"  Question:      {m.get('question', 'N/A')}")
    print(f"  Description:   {str(m.get('description', ''))[:120]}")
    print(f"  Condition ID:  {m.get('conditionId', 'N/A')}")
    print(f"  Question ID:   {m.get('questionId', 'N/A')}")
    print(f"  Active:        {m.get('active', 'N/A')}")
    print(f"  Volume:        ${float(m.get('volume', 0)):.2f}")
    print(f"  Liquidity:     ${float(m.get('liquidity', 0)):.2f}")

    # Tokens
    tokens = m.get("tokens", m.get("clobTokenIds", []))
    print(f"  UP token:      {tokens[0] if tokens else 'N/A'}")
    print(f"  DOWN token:    {tokens[1] if len(tokens) > 1 else 'N/A'}")

    # Resolution source
    res_source = (
        m.get("resolutionSource", "") or
        m.get("resolution_source", "") or
        "See description"
    )
    print(f"  Resolution:    {res_source}")

    # Fetch CLOB orderbook
    if tokens:
        up_token = tokens[0] if isinstance(tokens[0], str) else tokens[0].get("token_id", "")
        if up_token:
            print()
            print("  CLOB ORDERBOOK (UP token)")
            print("  " + "-" * 50)
            await _show_clob_book(session, up_token)

    print()
    print("  RESOLUTION RULES SUMMARY")
    print("  " + "-" * 50)
    print("  Oracle:        Chainlink BTC/USD (via Polymarket RTDS)")
    print("  RTDS WS:       wss://ws-live-data.polymarket.com")
    print("  Topic:         crypto_prices_chainlink")
    print("  Symbol:        btc/usd")
    print("  Open price:    First Chainlink tick at or after window open")
    print("  Resolution:    Chainlink close >= open → UP, else → DOWN")
    print("  Win fee:       2% of profit")
    print("  Taker fee:     ~0.5% of order value")
    print("=" * 60)


async def _show_clob_book(session: aiohttp.ClientSession, token_id: str) -> None:
    url = f"{POLYMARKET_CLOB_REST}/book?token_id={token_id}"
    try:
        async with session.get(url) as r:
            if r.status != 200:
                print(f"  CLOB error: {r.status}")
                return
            book = await r.json()
    except Exception as e:
        print(f"  CLOB error: {e}")
        return

    bids = sorted(book.get("bids", []), key=lambda x: -float(x["price"]))[:5]
    asks = sorted(book.get("asks", []), key=lambda x:  float(x["price"]))[:5]

    bid_mid = float(bids[0]["price"]) if bids else 0
    ask_mid = float(asks[0]["price"]) if asks else 1

    print(f"  Best bid: ${bid_mid:.4f}  Best ask: ${ask_mid:.4f}")
    print(f"  Mid:      ${(bid_mid + ask_mid) / 2:.4f}")
    print(f"  Spread:   ${ask_mid - bid_mid:.4f}")
    print()
    print("  Asks (offers to sell UP):")
    for a in asks[:3]:
        print(f"    ask ${float(a['price']):.4f}  size ${float(a['size']):.2f}")
    print("  Bids (offers to buy UP):")
    for b in bids[:3]:
        print(f"    bid ${float(b['price']):.4f}  size ${float(b['size']):.2f}")


async def _watch_chainlink() -> None:
    """Stream live Chainlink BTC/USD ticks."""
    import websockets

    sub = json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": json.dumps({"symbol": "btc/usd"}),
        }],
    })

    async with websockets.connect(POLYMARKET_RTDS_WS) as ws:
        await ws.send(sub)
        prev_price = 0.0
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            if msg.get("topic") != "crypto_prices_chainlink":
                continue
            payload = msg.get("payload", {})
            price = float(payload.get("value", 0))
            ts_ms = int(payload.get("timestamp", 0))
            delta = price - prev_price
            sign  = "▲" if delta > 0 else "▼" if delta < 0 else "="
            print(
                f"  [{_fmt_ts(ts_ms // 1000)}]  "
                f"Chainlink BTC/USD = ${price:,.2f}  "
                f"{sign}{abs(delta):.2f}"
            )
            prev_price = price


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%H:%M:%S UTC")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Polymarket BTC market")
    parser.add_argument("--watch", action="store_true",
                        help="Watch live Chainlink feed after inspection")
    args = parser.parse_args()
    asyncio.run(inspect_live(watch=args.watch))
