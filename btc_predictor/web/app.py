"""
Mobile-first web app for BTC Polymarket Predictor.

Serves a single-page dashboard at / and streams live data via SSE.

Endpoints:
  GET  /               → mobile HTML dashboard
  GET  /api/status     → current market + model status (JSON)
  GET  /api/prices     → recent predictions + fills (JSON)
  GET  /api/equity     → equity curve (JSON)
  GET  /stream/price   → Server-Sent Events: live Chainlink BTC/USD

Run:
  python -m btc_predictor.web.app
  python -m btc_predictor.web.app --port 8080 --host 0.0.0.0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import AsyncIterator

import aiohttp
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from btc_predictor.config import (
    MARKET_WINDOW_SECONDS,
    POLYMARKET_RTDS_WS,
    SQLITE_DB_PATH,
)

# -------------------------------------------------------------------------
# App setup
# -------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_chainlink_streamer())
    yield

app = FastAPI(title="BTC Polymarket Predictor", docs_url=None, redoc_url=None, lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Serve static files if any
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -------------------------------------------------------------------------
# In-memory state (populated by background task)
# -------------------------------------------------------------------------

_state: dict = {
    "chainlink_price": 0.0,
    "chainlink_ts_ms": 0,
    "price_history":   [],   # last 60 ticks: (ts_ms, price)
    "connected":       False,
}


# -------------------------------------------------------------------------
# Background: Chainlink price streamer
# -------------------------------------------------------------------------

async def _chainlink_streamer() -> None:
    """Connect to Polymarket RTDS and update _state continuously."""
    import websockets

    sub = json.dumps({
        "action": "subscribe",
        "subscriptions": [{
            "topic": "crypto_prices_chainlink",
            "type":  "*",
            "filters": json.dumps({"symbol": "btc/usd"}),
        }],
    })

    while True:
        try:
            async with websockets.connect(
                POLYMARKET_RTDS_WS, ping_interval=20, ping_timeout=30
            ) as ws:
                await ws.send(sub)
                _state["connected"] = True
                while True:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        continue

                    msg = json.loads(raw)
                    if msg.get("topic") != "crypto_prices_chainlink":
                        continue
                    p = msg.get("payload", {})
                    price = float(p.get("value", 0))
                    ts_ms = int(p.get("timestamp", 0))
                    if price > 0:
                        _state["chainlink_price"] = price
                        _state["chainlink_ts_ms"] = ts_ms
                        hist = _state["price_history"]
                        hist.append({"ts": ts_ms, "p": price})
                        if len(hist) > 300:
                            _state["price_history"] = hist[-300:]

        except Exception:
            _state["connected"] = False
            await asyncio.sleep(3)




# -------------------------------------------------------------------------
# SSE: stream live price to browser
# -------------------------------------------------------------------------

async def _price_event_stream() -> AsyncIterator[str]:
    last_ts = 0
    while True:
        ts = _state["chainlink_ts_ms"]
        if ts != last_ts and _state["chainlink_price"] > 0:
            data = json.dumps({
                "price":   _state["chainlink_price"],
                "ts_ms":   ts,
                "connected": _state["connected"],
            })
            yield f"data: {data}\n\n"
            last_ts = ts
        await asyncio.sleep(0.2)


@app.get("/stream/price")
async def stream_price():
    return StreamingResponse(
        _price_event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# -------------------------------------------------------------------------
# API: market status
# -------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    now = int(time.time())
    window_open  = now - (now % MARKET_WINDOW_SECONDS)
    window_close = window_open + MARKET_WINDOW_SECONDS
    elapsed      = now - window_open
    remaining    = window_close - now
    pct_done     = elapsed / MARKET_WINDOW_SECONDS

    # Read latest prediction from DB
    latest_pred = None
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM predictions ORDER BY ts_s DESC LIMIT 1"
            ).fetchone()
            if row:
                latest_pred = dict(row)
    except Exception:
        pass

    return {
        "chainlink_price": _state["chainlink_price"],
        "chainlink_ts_ms": _state["chainlink_ts_ms"],
        "feed_connected":  _state["connected"],
        "now_s":           now,
        "window_open_ts":  window_open,
        "window_close_ts": window_close,
        "elapsed_s":       elapsed,
        "remaining_s":     remaining,
        "pct_done":        round(pct_done, 4),
        "market_slug":     f"btc-updown-5m-{window_open}",
        "latest_prediction": latest_pred,
    }


# -------------------------------------------------------------------------
# API: predictions history
# -------------------------------------------------------------------------

@app.get("/api/predictions")
async def get_predictions(limit: int = 20):
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY ts_s DESC LIMIT ?", (limit,)
            ).fetchall()
            return {"predictions": [dict(r) for r in rows]}
    except Exception as e:
        return {"predictions": [], "error": str(e)}


# -------------------------------------------------------------------------
# API: equity curve
# -------------------------------------------------------------------------

@app.get("/api/equity")
async def get_equity(limit: int = 200):
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT ts_s, equity FROM equity_snapshots ORDER BY ts_s DESC LIMIT ?",
                (limit,),
            ).fetchall()
            rows = list(reversed(rows))
            return {"equity": [dict(r) for r in rows]}
    except Exception as e:
        return {"equity": [], "error": str(e)}


# -------------------------------------------------------------------------
# API: summary stats
# -------------------------------------------------------------------------

@app.get("/api/summary")
async def get_summary():
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row

            preds = conn.execute("SELECT * FROM predictions").fetchall()
            fills = conn.execute("SELECT pnl_net FROM fills").fetchall()

            n_total    = len(preds)
            resolved   = [p for p in preds if dict(p)["resolved_up"] is not None]
            n_resolved = len(resolved)

            total_pnl = sum(dict(f)["pnl_net"] or 0 for f in fills)
            hit_rate  = (
                sum(1 for p in resolved if (dict(p)["pnl_net"] or 0) > 0) / n_resolved
                if n_resolved > 0 else 0.0
            )

            # Rolling Brier (last 100)
            recent = resolved[-100:]
            brier  = 0.25
            if len(recent) >= 10:
                import math
                brier = sum(
                    (dict(p)["p_model"] - dict(p)["resolved_up"]) ** 2
                    for p in recent
                ) / len(recent)

            equity_row = conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY ts_s DESC LIMIT 1"
            ).fetchone()
            current_equity = dict(equity_row)["equity"] if equity_row else 10_000.0

        return {
            "n_predictions": n_total,
            "n_resolved":    n_resolved,
            "total_pnl":     round(total_pnl, 2),
            "hit_rate":      round(hit_rate, 4),
            "brier_score":   round(brier, 4),
            "current_equity": round(current_equity, 2),
        }
    except Exception as e:
        return {"error": str(e)}


# -------------------------------------------------------------------------
# Price history for chart
# -------------------------------------------------------------------------

@app.get("/api/price-history")
async def get_price_history():
    return {"history": _state["price_history"][-120:]}


# -------------------------------------------------------------------------
# Main HTML page
# -------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse(_INLINE_HTML)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BTC Polymarket web dashboard")
    parser.add_argument("--host",  default="0.0.0.0")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    local_ip = _get_local_ip()
    print(f"\n  BTC Polymarket Predictor Dashboard")
    print(f"  ====================================")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  Phone:   http://{local_ip}:{args.port}")
    print(f"  (both phone and computer must be on the same WiFi)\n")

    uvicorn.run(
        "btc_predictor.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )


def _get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


if __name__ == "__main__":
    main()


# -------------------------------------------------------------------------
# Inline HTML fallback (used if static/index.html doesn't exist)
# -------------------------------------------------------------------------

_INLINE_HTML = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>BTC Polymarket</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
</head><body style="font-family:monospace;padding:2rem;background:#111;color:#0f0">
<h2>BTC Polymarket Predictor</h2>
<p>Loading dashboard... if this persists, ensure static/index.html exists.</p>
<script>
fetch('/api/status').then(r=>r.json()).then(d=>{
  document.body.innerHTML += '<pre>' + JSON.stringify(d,null,2) + '</pre>';
});
</script>
</body></html>"""
