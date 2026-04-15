#!/usr/bin/env python3
"""
Historical data fetcher.

Downloads and stores:
  1. Binance BTC-USDT 1-second klines (spot)
  2. Chainlink BTC/USD prices via Polymarket RTDS (live recording)
  3. Polymarket CLOB orderbook snapshots via REST

For Chainlink history: we record live ticks starting from now.
There is no bulk historical Chainlink API; to backtest on real Chainlink
prices, run this script live for ≥30 days to build a dataset.

IMPORTANT: Binance 1s klines are only available for ~48 hours via REST.
For longer histories, use Binance Data Vision:
  https://data.binance.vision/?prefix=data/spot/daily/klines/BTCUSDT/1s/

Usage:
    python -m btc_predictor.scripts.fetch_history --days 1
    python -m btc_predictor.scripts.fetch_history --record-live --duration 3600
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from btc_predictor.config import (
    BINANCE_REST,
    BINANCE_SYMBOL,
    DATA_DIR,
    MARKET_SLUG_PREFIX,
    MARKET_WINDOW_SECONDS,
    POLYMARKET_CLOB_REST,
    POLYMARKET_GAMMA_REST,
    POLYMARKET_RTDS_WS,
)


async def fetch_binance_klines(
    date: str,
    output_dir: Path,
) -> None:
    """
    Download Binance 1-second klines for a given date from Binance Data Vision.
    Date format: 'YYYY-MM-DD'
    """
    url = (
        f"https://data.binance.vision/data/spot/daily/klines/"
        f"{BINANCE_SYMBOL}/1s/{BINANCE_SYMBOL}-1s-{date}.zip"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"bars_binance_{date.replace('-','')}.parquet"

    if parquet_path.exists():
        logger.info(f"[fetch] Already exists: {parquet_path}")
        return

    logger.info(f"[fetch] Downloading Binance 1s klines for {date}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as r:
            if r.status != 200:
                logger.warning(f"[fetch] Binance kline {date} not available: {r.status}")
                return
            data = await r.read()

    # Parse zip → CSV → parquet
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, header=None, names=[
                "open_time", "open", "high", "low", "close",
                "volume", "close_time", "quote_volume", "trade_count",
                "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
            ])

    df["ts_s"]       = df["open_time"] // 1000
    df["exchange"]   = "binance"
    df["buy_volume"] = df["taker_buy_base_vol"].astype(float)
    df["sell_volume"] = df["volume"].astype(float) - df["buy_volume"]
    df["midprice"]   = ((df["high"].astype(float) + df["low"].astype(float)) / 2)
    df["spread"]     = 0.0
    df["microprice"] = df["midprice"]

    cols = [
        "ts_s", "exchange", "open", "high", "low", "close",
        "volume", "buy_volume", "sell_volume", "trade_count",
        "midprice", "spread", "microprice",
    ]
    df = df[cols].astype({"open": float, "high": float, "low": float,
                           "close": float, "volume": float})
    df.to_parquet(parquet_path, index=False)
    logger.info(f"[fetch] Saved {len(df)} bars to {parquet_path}")


async def record_live_chainlink(
    output_dir: Path,
    duration_s: int = 3600,
) -> None:
    """
    Record live Chainlink BTC/USD ticks to daily parquet files.
    Runs until duration_s expires (or Ctrl-C).
    """
    import websockets

    output_dir.mkdir(parents=True, exist_ok=True)
    ticks = []
    start_time = time.time()

    sub = json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type": "*",
            "filters": json.dumps({"symbol": "btc/usd"}),
        }],
    })

    logger.info(f"[fetch] Recording Chainlink for {duration_s}s")
    async with websockets.connect(POLYMARKET_RTDS_WS) as ws:
        await ws.send(sub)
        while time.time() - start_time < duration_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                continue

            msg = json.loads(raw)
            if msg.get("topic") != "crypto_prices_chainlink":
                continue
            p = msg.get("payload", {})
            ticks.append({
                "ts_ms": int(p.get("timestamp", 0)),
                "price": float(p.get("value", 0)),
            })

    if ticks:
        df = pd.DataFrame(ticks)
        date_str = datetime.fromtimestamp(
            ticks[0]["ts_ms"] / 1000, tz=timezone.utc
        ).strftime("%Y%m%d")
        path = output_dir / f"chainlink_{date_str}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"[fetch] Saved {len(df)} Chainlink ticks to {path}")


async def snapshot_polymarket_books(
    output_dir: Path,
    duration_s: int = 3600,
    interval_s: int = 5,
) -> None:
    """
    Poll Polymarket CLOB orderbook snapshots every interval_s seconds.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    snaps = []
    start_time = time.time()

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        while time.time() - start_time < duration_s:
            now = int(time.time())
            open_ts = now - (now % MARKET_WINDOW_SECONDS)
            slug = f"{MARKET_SLUG_PREFIX}-{open_ts}"

            try:
                url = f"{POLYMARKET_GAMMA_REST}/markets?slug={slug}"
                async with session.get(url) as r:
                    data = await r.json()

                if data:
                    m = data[0] if isinstance(data, list) else data
                    tokens = m.get("tokens", m.get("clobTokenIds", []))
                    if len(tokens) >= 2:
                        up_id = tokens[0] if isinstance(tokens[0], str) else tokens[0].get("token_id", "")
                        book_url = f"{POLYMARKET_CLOB_REST}/book?token_id={up_id}"
                        async with session.get(book_url) as br:
                            book = await br.json()

                        bids = book.get("bids", [])
                        asks = book.get("asks", [])
                        best_bid = float(bids[0]["price"]) if bids else 0.0
                        best_ask = float(asks[0]["price"]) if asks else 1.0
                        mid = (best_bid + best_ask) / 2

                        snaps.append({
                            "ts_s": now,
                            "condition_id": m.get("conditionId", slug),
                            "up_bid": best_bid,
                            "up_ask": best_ask,
                            "up_mid": mid,
                            "spread": best_ask - best_bid,
                        })
            except Exception as e:
                logger.debug(f"[fetch] PM snapshot error: {e}")

            await asyncio.sleep(interval_s)

    if snaps:
        df = pd.DataFrame(snaps)
        date_str = datetime.fromtimestamp(snaps[0]["ts_s"], tz=timezone.utc).strftime("%Y%m%d")
        path = output_dir / f"polymarket_books_{date_str}.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"[fetch] Saved {len(df)} PM snapshots to {path}")


async def main(args) -> None:
    output_dir = Path(args.data_dir)

    if args.record_live:
        # Record all live feeds simultaneously
        await asyncio.gather(
            record_live_chainlink(output_dir, args.duration),
            snapshot_polymarket_books(output_dir, args.duration),
        )
    elif args.days:
        # Download historical Binance bars
        import pandas as pd
        end   = pd.Timestamp.utcnow().floor("D")
        start = end - pd.Timedelta(days=args.days)
        dates = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
        for date in dates:
            await fetch_binance_klines(date.strftime("%Y-%m-%d"), output_dir)
    else:
        logger.info("Use --days N to fetch Binance history, or --record-live to record live data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical BTC data")
    parser.add_argument("--days", type=int, default=0,
                        help="Download N days of Binance 1s klines")
    parser.add_argument("--record-live", action="store_true",
                        help="Record live Chainlink + Polymarket data")
    parser.add_argument("--duration", type=int, default=3600,
                        help="Duration in seconds for live recording (default: 1h)")
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Output directory (default: {DATA_DIR})")
    args = parser.parse_args()
    asyncio.run(main(args))
