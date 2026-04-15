"""
Central configuration for the BTC Polymarket predictor.

Resolution oracle: Chainlink BTC/USD via Polymarket RTDS
  wss://ws-live-data.polymarket.com  topic=crypto_prices_chainlink  symbol=btc/usd

The reference (open) price is the Chainlink price locked at market open.
The resolution price is the Chainlink price at the exact 5-min boundary.
Markets follow slug pattern: btc-updown-5m-{unix_ts_divisible_by_300}
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Network endpoints
# ---------------------------------------------------------------------------

POLYMARKET_CLOB_WS        = "wss://clob.polymarket.com"
POLYMARKET_CLOB_REST      = "https://clob.polymarket.com"
POLYMARKET_GAMMA_REST     = "https://gamma-api.polymarket.com"
POLYMARKET_RTDS_WS        = "wss://ws-live-data.polymarket.com"

BINANCE_WS                = "wss://stream.binance.com:9443/ws"
BINANCE_REST              = "https://api.binance.com"
COINBASE_WS               = "wss://advanced-trade-ws.coinbase.com"
BYBIT_WS                  = "wss://stream.bybit.com/v5/public/spot"

# ---------------------------------------------------------------------------
# Market parameters
# ---------------------------------------------------------------------------

MARKET_WINDOW_SECONDS     = 300          # 5 minutes
MARKET_SLUG_PREFIX        = "btc-updown-5m"
RESOLUTION_SYMBOL         = "btc/usd"   # Chainlink symbol key
BINANCE_SYMBOL            = "BTCUSDT"
COINBASE_SYMBOL           = "BTC-USD"
BYBIT_SYMBOL              = "BTCUSDT"

# Resolution: Chainlink BTC/USD at close epoch >= open epoch → UP
# Open price locked at the first Chainlink tick at or after the window open.
CHAINLINK_LEAD_ASSET      = "btc/usd"   # Chainlink RTDS topic filter

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

BAR_SECONDS               = 1           # 1-second bars
RETURN_HORIZONS_S         = [5, 30, 60, 180]   # log-return lookback windows
REALIZED_VOL_WINDOW_S     = 60          # rolling window for realized vol
OBI_LEVELS_SHALLOW        = 5           # top-N levels for shallow OBI
OBI_LEVELS_DEEP           = 20          # top-N levels for deep OBI
VPIN_BUCKET_SIZE          = 50          # trades per VPIN bucket
SEQUENCE_LENGTH_S         = 60          # lookback for temporal model (1D CNN)

# ---------------------------------------------------------------------------
# Model parameters
# ---------------------------------------------------------------------------

LGBM_PARAMS: dict = {
    "objective":       "binary",
    "metric":          "binary_logloss",
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":    5,
    "lambda_l1":       0.1,
    "lambda_l2":       0.1,
    "n_estimators":    500,
    "early_stopping_rounds": 30,
    "verbose":         -1,
    "n_jobs":          -1,
}

CNN_HIDDEN_CHANNELS       = 64
CNN_KERNEL_SIZE           = 5
CNN_NUM_LAYERS            = 3
CNN_DROPOUT               = 0.2
CNN_LR                    = 1e-3
CNN_EPOCHS                = 50
CNN_BATCH_SIZE            = 256

ENSEMBLE_LGBM_WEIGHT      = 0.6
ENSEMBLE_CNN_WEIGHT       = 0.4

# Purged k-fold
PURGED_KFOLD_N_SPLITS     = 5
PURGED_EMBARGO_S          = 60          # embargo gap in seconds

# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------

POLYMARKET_WIN_FEE        = 0.02        # 2% fee on winnings
POLYMARKET_TAKER_FEE      = 0.005       # 0.5% taker fee (added late 2025)
SLIPPAGE_TICKS            = 2           # assumed price impact in ticks
TICK_SIZE                 = 0.001       # Polymarket minimum tick
MIN_LIQUIDITY_USDC        = 50.0        # skip if less than $50 on best bid/ask
MAX_POSITION_USDC         = 100.0       # max bet per market
MIN_EDGE                  = 0.03        # minimum p - m to trade (after fees)

# ---------------------------------------------------------------------------
# Live execution
# ---------------------------------------------------------------------------

KELLY_FRACTION            = 0.25        # fractional Kelly cap
MAX_KELLY_BET_USDC        = 100.0
SAFETY_MARGIN             = 0.01        # extra buffer beyond fees+slippage
STALE_DATA_THRESHOLD_S    = 2.0         # halt if no Chainlink tick in 2s
MAX_DRAWDOWN_PCT          = 0.20        # kill switch: 20% drawdown
MODEL_PROB_BOUNDS         = (0.01, 0.99) # kill if outside these bounds
ENTRY_CUTOFF_S            = 120         # don't enter within 2 min of close

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------

SQLITE_DB_PATH            = os.environ.get(
    "SQLITE_DB_PATH", "data/monitoring.db"
)
DASHBOARD_PORT            = int(os.environ.get("DASHBOARD_PORT", "8501"))
ROLLING_BRIER_WINDOW      = 100         # last N predictions for rolling Brier

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR                  = os.environ.get("DATA_DIR", "data/raw")
MODEL_DIR                 = os.environ.get("MODEL_DIR", "models/saved")
LOG_DIR                   = os.environ.get("LOG_DIR", "logs")


@dataclass
class LiveConfig:
    """Runtime-overridable live trading configuration."""
    dry_run: bool = True
    kelly_fraction: float = KELLY_FRACTION
    max_bet_usdc: float = MAX_KELLY_BET_USDC
    min_edge: float = MIN_EDGE
    safety_margin: float = SAFETY_MARGIN
    max_drawdown_pct: float = MAX_DRAWDOWN_PCT
    stale_data_threshold_s: float = STALE_DATA_THRESHOLD_S
    entry_cutoff_s: int = ENTRY_CUTOFF_S
    verbose: bool = True
