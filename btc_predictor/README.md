# BTC Polymarket Predictor

Production-grade 5-minute BTC Up/Down predictor for Polymarket's recurring binary markets.
Optimized for **risk-adjusted PnL against implied probability**, not raw accuracy.

## Edge Hypothesis

### The Oracle Lag

Polymarket's 5-minute BTC markets resolve using **Chainlink's BTC/USD Data Streams** (not raw Binance prices). Chainlink is an aggregated oracle that pulls from multiple venues — it inherits a systematic latency of **100ms–2s** behind the Binance spot price.

This creates a predictable relationship:
```
oracle_lead = binance_mid − chainlink_price
```

When Binance has made a sustained directional move significantly above (or below) the Chainlink oracle, it strongly predicts the direction the Chainlink resolution price will settle. This is the **primary edge feature**.

### Secondary signals
- **Order book imbalance** (OBI) at top-5/top-20 levels: informed traders leave footprints
- **Trade flow imbalance** + **VPIN**: distinguishes noise from informed order flow
- **Cross-venue OBI spread**: divergence between Binance and Coinbase signals institutional flow
- **Vol regime** (vol-of-vol, vol ratio): markets with expanding vol are more directional
- **Time into window** + **open gap**: late-window entries have higher resolution signal

### Why calibration > accuracy

A 55% accurate but **miscalibrated** model will lose money. When it says "70% UP," you bet large — but if 70% signals only win 55% of the time, you're over-betting and the variance destroys you. Isotonic regression on a held-out calibration set ensures the probability outputs are trustworthy.

## Architecture

```
btc_predictor/
├── data/               Async WebSocket + REST collectors
│   ├── chainlink_feed.py   Polymarket RTDS → Chainlink BTC/USD (RESOLUTION ORACLE)
│   ├── binance_feed.py     Binance BTC-USDT trades + L2 orderbook
│   ├── coinbase_feed.py    Coinbase BTC-USD trades + orderbook
│   ├── bybit_feed.py       Bybit BTC-USDT trades + orderbook
│   ├── polymarket_feed.py  Polymarket CLOB orderbook + market discovery
│   ├── funding_rates.py    Perpetual funding rates (Binance + Bybit)
│   ├── cme_basis.py        CME futures basis (via Binance delivery)
│   └── bar_builder.py      1-second OHLCV bar aggregator
├── features/           Feature engineering (no look-ahead by design)
│   ├── pipeline.py         Main pipeline → FeatureVector
│   ├── returns.py          Log returns at 5s/30s/1m/3m
│   ├── volatility.py       Realized vol, vol-of-vol, vol ratio
│   ├── orderbook.py        OBI shallow/deep, microprice, spread, depth
│   └── flow.py             TFI, volume momentum, VPIN
├── models/             ML models
│   ├── lgbm_model.py       LightGBM binary classifier
│   ├── temporal_model.py   1D CNN over 60-second feature sequences
│   ├── ensemble.py         Weighted blend + isotonic calibration
│   ├── cv.py               Purged k-fold CV with embargo
│   └── calibration.py      Isotonic regression calibrator
├── backtest/           Event-driven backtester
│   ├── engine.py           Replays historical data with realistic fills
│   ├── fills.py            Fee model (2% win fee + 0.5% taker fee + slippage)
│   └── report.py           Sharpe, max DD, hit rate, edge vs market
├── live/               Live execution
│   ├── executor.py         1-Hz decision loop
│   ├── sizing.py           Fractional Kelly criterion
│   └── killswitch.py       5 kill-switch conditions
└── monitoring/         Observability
    ├── logger.py           SQLite: predictions, fills, PnL
    └── dashboard.py        Streamlit: calibration curve, Brier, PnL
```

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Inspect the live market oracle FIRST (verify resolution rules)
make inspect

# 3. Collect data (Chainlink must be recorded live; no bulk API exists)
make record          # 1 hour of live Chainlink + Polymarket data
make fetch-history   # 30 days of Binance 1s klines

# 4. After collecting ≥30 days of Chainlink data, train
make train

# 5. Backtest on historical data
make backtest

# 6. Run live in dry-run mode
make live

# 7. Monitor via dashboard
make dashboard       # http://localhost:8501
```

## Resolution Rules (Confirmed)

| Parameter | Value |
|-----------|-------|
| Oracle | **Chainlink BTC/USD** via Polymarket RTDS |
| RTDS WebSocket | `wss://ws-live-data.polymarket.com` |
| Topic | `crypto_prices_chainlink` |
| Symbol | `btc/usd` |
| Open price | First Chainlink tick at or after the 300s window boundary |
| Resolution | close ≥ open → UP; close < open → DOWN |
| Win fee | 2% of net profit |
| Taker fee | ~0.5% of order value |
| Market slug | `btc-updown-5m-{unix_ts}` (unix_ts divisible by 300) |

> **Critical**: Train and predict on the Chainlink price, not Binance BTCUSDT.
> The basis risk between them is the *signal*, not the noise.

## Fee Model & Break-Even

For a bet of $X on UP at price p:
- Cost: $X (buys X/p shares)  
- Win: receive X/p × $1.00
- Win fee: 2% × (X/p - X)
- Taker fee: 0.5% × X (paid at entry)
- Net win: (X/p - X) × 0.98 - 0.005X

**Break-even probability:**
```
q_be = (1 + taker_fee) / ((1/p - 1) × (1 - win_fee) + 1)
```

At p=0.50: q_be ≈ 0.514 — you need >51.4% accuracy just to break even.

## Kill Switches

Trading halts automatically on:

| Condition | Threshold |
|-----------|-----------|
| Stale Chainlink data | > 2 seconds since last tick |
| Drawdown | > 20% from peak equity |
| Model probability | Outside [0.01, 0.99] |
| Exchange disconnect | WebSocket retry budget exhausted |
| Manual halt | Operator sets flag via dashboard |

## Known Risks & Limitations

1. **Chainlink history unavailable**: Chainlink prices must be recorded live via RTDS. No bulk historical API exists. This means the model needs ≥30 days of live recording before training.

2. **Oracle lead is not guaranteed**: Chainlink's lag behind Binance is empirical and may narrow or disappear with infrastructure changes.

3. **Market efficiency**: Other participants likely use similar oracle-lag strategies. As markets become efficient, edge erodes. Monitor Brier score rolling to detect degradation.

4. **Liquidity constraints**: At large bet sizes, you move the Polymarket CLOB and erode your own edge. The $100 cap is conservative.

5. **Regulatory risk**: Prediction market access varies by jurisdiction. Verify legal status before depositing.

6. **Execution risk**: Live order submission requires implementing `executor.py:_submit_order` with Polymarket API credentials (`py-clob-client`).

7. **No Polymarket orderbook history**: Polymarket CLOB snapshots must also be recorded live. The backtest uses synthetic orderbooks when PM snapshots are unavailable.

## Backtest Report (Template)

After running `make backtest` on ≥30 days of data, you should see:

```
=======================================================
  BTC Polymarket Backtest Report
=======================================================
  Period:           30 days
  Markets seen:     8640   (288/day × 30 days)
  Trades:           N      (markets where edge threshold met)
  ...
  Sharpe ratio:     ?      (target: >1.5)
  Max drawdown:     ?%     (target: <15%)
  Hit rate:         ?      (target: >52%)
  Edge vs 50%:      ?      (target: >2%)
=======================================================
```

> **Note**: Realistic backtest results require real Chainlink history.
> Binance-only backtests will overstate performance due to perfect oracle data.

## Latency Budget

| Step | Target | Notes |
|------|--------|-------|
| Feature compute | < 50ms | Vectorized numpy ops |
| LightGBM inference | < 10ms | Single row predict |
| CNN inference | < 50ms | 60×F matrix on CPU |
| Total | **< 200ms** | Profiled in ensemble.py |

## Development

```bash
# Run all tests
make test

# Critical: no-lookahead tests MUST pass
make test-lookahead

# Lint
make lint
```
