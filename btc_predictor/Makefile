# BTC Polymarket Predictor — Makefile
# ====================================
# make install       — install dependencies
# make inspect       — inspect live Polymarket market + oracle source
# make record        — record 1 hour of live Chainlink + PM data
# make fetch-history — download 30 days of Binance 1s klines
# make train         — train the ensemble model
# make backtest      — run backtest on saved historical data
# make test          — run all tests (including no-lookahead tests)
# make live          — start live executor (dry-run by default)
# make dashboard     — launch Streamlit monitoring dashboard
# make lint          — run ruff linter

PYTHON := python3
DATA_DIR := data/raw
MODEL_DIR := models/saved
LOG_DIR := logs
DB_PATH := data/monitoring.db

.PHONY: all install inspect record fetch-history train backtest test live dashboard lint clean

all: install

install:
	$(PYTHON) -m pip install -r btc_predictor/requirements.txt

# -----------------------------------------------------------------------
# Market inspection — ALWAYS run this before training to verify oracle
# -----------------------------------------------------------------------
inspect:
	$(PYTHON) -m btc_predictor.scripts.inspect_market

inspect-watch:
	$(PYTHON) -m btc_predictor.scripts.inspect_market --watch

# -----------------------------------------------------------------------
# Data collection
# -----------------------------------------------------------------------
record:
	@mkdir -p $(DATA_DIR)
	$(PYTHON) -m btc_predictor.scripts.fetch_history \
		--record-live \
		--duration 3600 \
		--data-dir $(DATA_DIR)

record-24h:
	@mkdir -p $(DATA_DIR)
	$(PYTHON) -m btc_predictor.scripts.fetch_history \
		--record-live \
		--duration 86400 \
		--data-dir $(DATA_DIR)

fetch-history:
	@mkdir -p $(DATA_DIR)
	$(PYTHON) -m btc_predictor.scripts.fetch_history \
		--days 30 \
		--data-dir $(DATA_DIR)
	@echo ""
	@echo "NOTE: Binance klines downloaded. For Chainlink price history,"
	@echo "      run 'make record' to capture live data before training."

# -----------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------
train:
	@mkdir -p $(MODEL_DIR) $(LOG_DIR)
	$(PYTHON) -m btc_predictor.scripts.train \
		--data-dir $(DATA_DIR)

train-custom:
	@mkdir -p $(MODEL_DIR) $(LOG_DIR)
	$(PYTHON) -m btc_predictor.scripts.train \
		--start $(START) \
		--end $(END) \
		--data-dir $(DATA_DIR)

# -----------------------------------------------------------------------
# Backtest
# -----------------------------------------------------------------------
backtest:
	@mkdir -p $(LOG_DIR)
	$(PYTHON) -c "\
import sys; sys.path.insert(0,'.')\n\
from btc_predictor.backtest.engine import BacktestEngine\n\
from btc_predictor.models.ensemble import EnsemblePredictor\n\
import pandas as pd\n\
model = EnsemblePredictor()\n\
model.load()\n\
end   = pd.Timestamp.utcnow().strftime('%Y-%m-%d')\n\
start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime('%Y-%m-%d')\n\
engine = BacktestEngine(model, start, end, data_dir='$(DATA_DIR)')\n\
report = engine.run()\n\
report.print_summary()\n\
df = report.to_dataframe()\n\
df.to_csv('$(LOG_DIR)/backtest_results.csv', index=False)\n\
print('Results saved to $(LOG_DIR)/backtest_results.csv')\n\
"

# -----------------------------------------------------------------------
# Live trading
# -----------------------------------------------------------------------
live:
	$(PYTHON) -c "\
import asyncio, sys; sys.path.insert(0,'.')\n\
from btc_predictor.live.executor import LiveExecutor\n\
from btc_predictor.config import LiveConfig\n\
cfg = LiveConfig(dry_run=True, verbose=True)\n\
asyncio.run(LiveExecutor(cfg).run())\n\
"

live-trade:
	@echo "WARNING: This will place real orders on Polymarket!"
	@echo "Requires POLYMARKET_PRIVATE_KEY and POLYMARKET_API_KEY env vars."
	@read -p "Are you sure? (yes/no): " confirm && [ "$$confirm" = "yes" ]
	$(PYTHON) -c "\
import asyncio, sys; sys.path.insert(0,'.')\n\
from btc_predictor.live.executor import LiveExecutor\n\
from btc_predictor.config import LiveConfig\n\
cfg = LiveConfig(dry_run=False, verbose=True)\n\
asyncio.run(LiveExecutor(cfg).run())\n\
"

# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------
test:
	$(PYTHON) -m pytest btc_predictor/tests/ -v --tb=short

test-lookahead:
	@echo "Running no-lookahead tests (these MUST all pass before training)..."
	$(PYTHON) -m pytest btc_predictor/tests/test_no_lookahead.py -v

test-fast:
	$(PYTHON) -m pytest btc_predictor/tests/ -x -q

# -----------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------
dashboard:
	streamlit run btc_predictor/monitoring/dashboard.py \
		--server.port 8501 \
		--server.address 0.0.0.0

# -----------------------------------------------------------------------
# Code quality
# -----------------------------------------------------------------------
lint:
	$(PYTHON) -m ruff check btc_predictor/ || true

# -----------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null; true
