#!/usr/bin/env python3
"""
Train the ensemble model on historical data.

Steps:
  1. Load parquet files from DATA_DIR
  2. Build feature matrix (flat + sequences)
  3. Construct labels: sign(chainlink_close - chainlink_open) per window
  4. Train LGBM + CNN ensemble with purged k-fold CV
  5. Calibrate with isotonic regression on holdout
  6. Save model to MODEL_DIR
  7. Print backtest report

Usage:
    python -m btc_predictor.scripts.train
    python -m btc_predictor.scripts.train --start 2025-01-01 --end 2025-03-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from btc_predictor.config import DATA_DIR, MARKET_WINDOW_SECONDS, SEQUENCE_LENGTH_S
from btc_predictor.data.bar_builder import Bar
from btc_predictor.features.returns import compute_return_features, compute_acceleration
from btc_predictor.features.volatility import compute_volatility_features
from btc_predictor.features.orderbook import compute_orderbook_features
from btc_predictor.features.flow import VPINCalculator, compute_flow_features
from btc_predictor.models.ensemble import EnsemblePredictor


def load_and_label(
    data_dir: Path,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.Series]:
    """
    Load data, build features, and create labels.

    Label: 1 if Chainlink close >= Chainlink open for the 5-min window, else 0.
    Feature timestamp: the second at which we'd make the prediction
    (entry_cutoff_s = 120s before close, so 180s into the window).
    """
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts   = pd.Timestamp(end,   tz="UTC")

    bars_parts, cl_parts = [], []
    current = start_ts
    while current <= end_ts:
        ds = current.strftime("%Y%m%d")
        bf = data_dir / f"bars_binance_{ds}.parquet"
        cf = data_dir / f"chainlink_{ds}.parquet"
        if bf.exists():
            bars_parts.append(pd.read_parquet(bf))
        if cf.exists():
            cl_parts.append(pd.read_parquet(cf))
        current += pd.Timedelta(days=1)

    if not bars_parts or not cl_parts:
        raise ValueError(
            f"No data found in {data_dir}.\n"
            "Run: python -m btc_predictor.scripts.fetch_history --days 30\n"
            "And record live Chainlink: python -m btc_predictor.scripts.fetch_history --record-live"
        )

    bars_df = pd.concat(bars_parts, ignore_index=True).sort_values("ts_s")
    cl_df   = pd.concat(cl_parts,   ignore_index=True).sort_values("ts_ms")
    cl_df["ts_s"] = (cl_df["ts_ms"] / 1000).astype(int)

    logger.info(f"Loaded {len(bars_df)} bars, {len(cl_df)} Chainlink ticks")

    # Build 5-min windows
    first_s = int(cl_df["ts_s"].min())
    last_s  = int(cl_df["ts_s"].max())

    rows       = []
    seqs       = []
    labels     = []
    timestamps = []

    window_starts = range(
        first_s - (first_s % MARKET_WINDOW_SECONDS),
        last_s - MARKET_WINDOW_SECONDS,
        MARKET_WINDOW_SECONDS,
    )

    # Stateful VPIN
    vpin = VPINCalculator()

    for open_ts in window_starts:
        close_ts = open_ts + MARKET_WINDOW_SECONDS

        # Chainlink open/close prices
        cl_window = cl_df[cl_df["ts_s"].between(open_ts, close_ts)]
        if cl_window.empty:
            continue
        cl_open  = float(cl_window.iloc[0]["price"])
        cl_after = cl_df[cl_df["ts_s"] >= close_ts]
        if cl_after.empty:
            continue
        cl_close = float(cl_after.iloc[0]["price"])
        label = int(cl_close >= cl_open)

        # Feature time: 180s into window (120s before close = entry point)
        feat_ts = open_ts + 180

        # Bars strictly before feat_ts
        bar_rows = bars_df[
            (bars_df["ts_s"] >= feat_ts - 200) &
            (bars_df["ts_s"] < feat_ts)
        ]
        if len(bar_rows) < 30:
            continue

        bars = [_row_to_bar(r) for _, r in bar_rows.iterrows()]

        # Update VPIN
        for b in bars:
            vpin.update(b.buy_volume, b.sell_volume)

        # Chainlink price at feat_ts
        cl_at_feat = cl_df[cl_df["ts_s"] < feat_ts]
        cl_price = float(cl_at_feat.iloc[-1]["price"]) if not cl_at_feat.empty else 0.0
        binance_mid = bars[-1].midprice if bars else 0.0
        oracle_lead = binance_mid - cl_price if cl_price > 0 else 0.0

        # Feature dict
        feats = {}
        feats.update(compute_return_features(bars, feat_ts))
        feats.update(compute_acceleration(bars, feat_ts))
        feats.update(compute_volatility_features(bars, feat_ts))
        ob = compute_orderbook_features(bars[-1] if bars else None)
        feats.update({f"bnb_{k}": v for k, v in ob.items()})
        feats.update({f"cb_{k}":  v for k, v in ob.items()})
        feats["cross_obi"] = 0.0
        feats.update(compute_flow_features(bars, feat_ts, vpin))
        feats["oracle_lead"]     = oracle_lead
        feats["oracle_lead_bps"] = oracle_lead / cl_price * 10_000 if cl_price > 0 else 0.0

        # Polymarket features (synthetic for training)
        feats["pm_implied_up"]     = 0.5
        feats["pm_spread"]         = 0.02
        feats["pm_time_remaining"] = float(close_ts - feat_ts)
        feats["pm_time_frac"]      = 1.0 - (close_ts - feat_ts) / MARKET_WINDOW_SECONDS
        feats["pm_open_gap_bps"]   = oracle_lead / cl_open * 10_000 if cl_open > 0 else 0.0
        feats["pm_book_depth_up"]  = 0.0
        feats["funding_binance"]   = 0.0
        feats["funding_bybit"]     = 0.0
        feats["cme_basis_ann"]     = 0.0

        # Sequence
        seq_bars = [b for b in bars if b.ts_s >= feat_ts - SEQUENCE_LENGTH_S]
        seq = None
        if len(seq_bars) >= SEQUENCE_LENGTH_S:
            keys = sorted(feats.keys())
            mat = []
            for b in seq_bars[-SEQUENCE_LENGTH_S:]:
                row_f = {k: feats.get(k, 0.0) for k in keys}
                mat.append([row_f[k] for k in keys])
            seq = np.array(mat, dtype=np.float32)
        else:
            seq = np.full((SEQUENCE_LENGTH_S, len(feats)), np.nan, dtype=np.float32)

        rows.append(feats)
        seqs.append(seq)
        labels.append(label)
        timestamps.append(feat_ts)

    logger.info(f"Built {len(rows)} training samples")
    X_flat     = pd.DataFrame(rows).fillna(0)
    X_seq      = np.stack(seqs)
    y          = pd.Series(labels)
    ts         = pd.Series(timestamps)
    return X_flat, X_seq, y, ts


def _row_to_bar(row) -> Bar:
    b = Bar(
        ts_s=int(row["ts_s"]),
        exchange="binance",
        open=float(row.get("open", 0)),
        high=float(row.get("high", 0)),
        low=float(row.get("low", 0)),
        close=float(row.get("close", 0)),
        volume=float(row.get("volume", 0)),
        buy_volume=float(row.get("buy_volume", 0)),
        sell_volume=float(row.get("sell_volume", 0)),
        trade_count=int(row.get("trade_count", 0)),
    )
    b.midprice = float(row.get("midprice", b.close))
    return b


def main(args) -> None:
    data_dir = Path(args.data_dir)
    logger.info(f"Training on {args.start} → {args.end}")

    X_flat, X_seq, y, ts = load_and_label(data_dir, args.start, args.end)

    logger.info(f"Dataset: {len(y)} samples, {X_flat.shape[1]} features")
    logger.info(f"Class balance: {y.mean():.3f} (fraction UP)")

    model = EnsemblePredictor()
    metrics = model.train(X_flat, X_seq, y, ts)
    model.save()

    print("\n--- Training Complete ---")
    print(f"Brier score (raw):        {metrics['brier_raw']:.4f}")
    print(f"Brier score (calibrated): {metrics['brier_cal']:.4f}")
    print(f"Accuracy:                 {metrics['accuracy']:.3f}")
    print("\nTop 15 feature importances:")
    for feat, imp in list(metrics["feature_importance"].items())[:15]:
        print(f"  {feat:<35} {imp:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ensemble model")
    import pandas as pd
    default_end   = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    default_start = (pd.Timestamp.utcnow() - pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    parser.add_argument("--start",    default=default_start)
    parser.add_argument("--end",      default=default_end)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()
    main(args)
