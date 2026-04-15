"""
SQLite logger for predictions, fills, and PnL.

Schema:
  predictions(id, ts_s, condition_id, p_model, market_mid, edge,
              direction, bet_usdc, resolved_up, pnl_net, created_at)
  fills(id, ts_s, condition_id, direction, entry_price, shares,
        usdc_spent, taker_fee, slippage_cost, resolved_up, pnl_gross,
        pnl_net, created_at)
  equity_snapshots(id, ts_s, equity, created_at)
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional

from btc_predictor.config import SQLITE_DB_PATH


CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_s        INTEGER NOT NULL,
    condition_id TEXT NOT NULL,
    p_model     REAL NOT NULL,
    market_mid  REAL NOT NULL,
    edge        REAL NOT NULL,
    direction   TEXT NOT NULL,
    bet_usdc    REAL NOT NULL,
    resolved_up INTEGER,
    pnl_net     REAL,
    created_at  INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"""

CREATE_FILLS = """
CREATE TABLE IF NOT EXISTS fills (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_s           INTEGER NOT NULL,
    condition_id   TEXT NOT NULL,
    direction      TEXT NOT NULL,
    entry_price    REAL NOT NULL,
    shares         REAL NOT NULL,
    usdc_spent     REAL NOT NULL,
    taker_fee      REAL NOT NULL,
    slippage_cost  REAL NOT NULL,
    resolved_up    INTEGER,
    pnl_gross      REAL,
    pnl_net        REAL,
    created_at     INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"""

CREATE_EQUITY = """
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_s       INTEGER NOT NULL,
    equity     REAL NOT NULL,
    created_at INTEGER NOT NULL DEFAULT (strftime('%s','now'))
);
"""


class TradeLogger:
    def __init__(self, db_path: str = SQLITE_DB_PATH) -> None:
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._db_path)

    def init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(CREATE_PREDICTIONS)
            conn.execute(CREATE_FILLS)
            conn.execute(CREATE_EQUITY)
            conn.commit()

    def log_prediction(
        self,
        ts_s: int,
        condition_id: str,
        p_model: float,
        market_mid: float,
        edge: float,
        direction: str,
        bet_usdc: float,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO predictions
                   (ts_s, condition_id, p_model, market_mid, edge, direction, bet_usdc)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (ts_s, condition_id, p_model, market_mid, edge, direction, bet_usdc),
            )
            conn.commit()
            return cur.lastrowid

    def update_prediction_outcome(
        self,
        condition_id: str,
        resolved_up: bool,
        pnl_net: float,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """UPDATE predictions SET resolved_up=?, pnl_net=?
                   WHERE condition_id=? AND resolved_up IS NULL""",
                (int(resolved_up), pnl_net, condition_id),
            )
            conn.commit()

    def log_fill(self, fill: "Fill") -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO fills
                   (ts_s, condition_id, direction, entry_price, shares,
                    usdc_spent, taker_fee, slippage_cost, resolved_up,
                    pnl_gross, pnl_net)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    fill.ts_s, fill.condition_id, fill.direction,
                    fill.entry_price, fill.shares, fill.usdc_spent,
                    fill.taker_fee, fill.slippage_cost,
                    int(fill.resolved_up) if fill.resolved_up is not None else None,
                    fill.pnl_gross, fill.pnl_net,
                ),
            )
            conn.commit()

    def log_equity(self, equity: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO equity_snapshots (ts_s, equity) VALUES (?, ?)",
                (int(time.time()), equity),
            )
            conn.commit()

    def get_current_equity(self, initial: float = 10_000.0) -> float:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT equity FROM equity_snapshots ORDER BY ts_s DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else initial

    def get_predictions_df(self):
        import pandas as pd
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM predictions ORDER BY ts_s", conn)

    def get_fills_df(self):
        import pandas as pd
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM fills ORDER BY ts_s", conn)

    def get_equity_df(self):
        import pandas as pd
        with self._conn() as conn:
            return pd.read_sql("SELECT * FROM equity_snapshots ORDER BY ts_s", conn)
