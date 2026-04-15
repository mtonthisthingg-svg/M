"""
Streamlit monitoring dashboard.

Displays:
  1. Live calibration curve (reliability diagram)
  2. Rolling Brier score (last N predictions)
  3. Cumulative PnL chart
  4. Feature importance (from trained model)
  5. Kill switch status
  6. Recent predictions table

Run: streamlit run btc_predictor/monitoring/dashboard.py --server.port 8501
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from btc_predictor.config import DASHBOARD_PORT, ROLLING_BRIER_WINDOW, SQLITE_DB_PATH
from btc_predictor.monitoring.logger import TradeLogger
from btc_predictor.models.calibration import IsotonicCalibrator

st.set_page_config(
    page_title="BTC Polymarket Monitor",
    page_icon="📊",
    layout="wide",
)

st.title("BTC Polymarket Predictor — Live Dashboard")


@st.cache_data(ttl=10)
def load_data():
    logger = TradeLogger(SQLITE_DB_PATH)
    preds  = logger.get_predictions_df()
    fills  = logger.get_fills_df()
    equity = logger.get_equity_df()
    return preds, fills, equity


preds_df, fills_df, equity_df = load_data()

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------
col1, col2, col3, col4, col5 = st.columns(5)

resolved = preds_df[preds_df["resolved_up"].notna()]
n_total   = len(preds_df)
n_resolved = len(resolved)

with col1:
    st.metric("Total Predictions", n_total)

with col2:
    hit_rate = float((resolved["pnl_net"] > 0).mean()) if n_resolved > 0 else 0.0
    st.metric("Hit Rate", f"{hit_rate:.1%}")

with col3:
    total_pnl = float(fills_df["pnl_net"].sum()) if not fills_df.empty else 0.0
    st.metric("Total PnL", f"${total_pnl:.2f}", delta_color="normal")

with col4:
    current_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else 0.0
    st.metric("Current Equity", f"${current_equity:.2f}")

with col5:
    if n_resolved >= 10:
        recent = resolved.tail(ROLLING_BRIER_WINDOW)
        brier = float(np.mean((recent["p_model"].values - recent["resolved_up"].values) ** 2))
        brier_ref = 0.25
        st.metric(
            f"Rolling Brier ({ROLLING_BRIER_WINDOW})",
            f"{brier:.4f}",
            delta=f"{brier - brier_ref:.4f} vs no-skill",
            delta_color="inverse",
        )
    else:
        st.metric("Rolling Brier", "—")

st.divider()

# ---------------------------------------------------------------------------
# Row 1: Calibration curve + Equity curve
# ---------------------------------------------------------------------------
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Calibration Curve (Reliability Diagram)")
    if n_resolved >= 20:
        probs  = resolved["p_model"].values.astype(float)
        labels = resolved["resolved_up"].values.astype(float)
        cal = IsotonicCalibrator()
        mean_pred, frac_pos, counts = cal.reliability_diagram(probs, labels, n_bins=10)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Perfect calibration",
        ))
        fig.add_trace(go.Scatter(
            x=mean_pred, y=frac_pos,
            mode="lines+markers",
            marker=dict(size=[max(5, c // 5) for c in counts]),
            name="Model",
            text=[f"n={c}" for c in counts],
            hovertemplate="%{text}<br>predicted=%{x:.2f}<br>actual=%{y:.2f}",
        ))
        fig.update_layout(
            xaxis_title="Mean predicted probability",
            yaxis_title="Fraction of positives",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Need ≥20 resolved predictions (have {n_resolved})")

with col_right:
    st.subheader("Cumulative PnL")
    if not fills_df.empty:
        fills_df = fills_df.sort_values("ts_s")
        fills_df["cum_pnl"] = fills_df["pnl_net"].cumsum()
        fills_df["datetime"] = pd.to_datetime(fills_df["ts_s"], unit="s")
        fig2 = px.line(
            fills_df, x="datetime", y="cum_pnl",
            labels={"cum_pnl": "Cumulative PnL ($)", "datetime": "Time"},
            height=350,
        )
        fig2.add_hline(y=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No fills yet")

st.divider()

# ---------------------------------------------------------------------------
# Row 2: Rolling Brier + Edge histogram
# ---------------------------------------------------------------------------
col_b, col_e = st.columns(2)

with col_b:
    st.subheader(f"Rolling Brier Score (window={ROLLING_BRIER_WINDOW})")
    if n_resolved >= ROLLING_BRIER_WINDOW:
        brier_vals = []
        for i in range(ROLLING_BRIER_WINDOW, n_resolved + 1):
            window = resolved.iloc[i - ROLLING_BRIER_WINDOW:i]
            bs = float(np.mean(
                (window["p_model"].values - window["resolved_up"].values) ** 2
            ))
            brier_vals.append({
                "ts_s": window.iloc[-1]["ts_s"],
                "brier": bs,
            })
        brier_df = pd.DataFrame(brier_vals)
        brier_df["datetime"] = pd.to_datetime(brier_df["ts_s"], unit="s")
        fig3 = px.line(
            brier_df, x="datetime", y="brier",
            labels={"brier": "Brier Score", "datetime": "Time"},
            height=300,
        )
        fig3.add_hline(y=0.25, line_dash="dash", line_color="red",
                       annotation_text="No-skill (0.25)")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(f"Need ≥{ROLLING_BRIER_WINDOW} resolved predictions")

with col_e:
    st.subheader("Edge Distribution")
    if not preds_df.empty:
        fig4 = px.histogram(
            preds_df, x="edge", nbins=30,
            labels={"edge": "Model Edge (p - market_mid)"},
            height=300,
        )
        fig4.add_vline(x=0, line_dash="dash", line_color="grey")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No predictions yet")

st.divider()

# ---------------------------------------------------------------------------
# Recent predictions table
# ---------------------------------------------------------------------------
st.subheader("Recent Predictions")
if not preds_df.empty:
    display_df = preds_df.tail(20).copy()
    display_df["datetime"] = pd.to_datetime(display_df["ts_s"], unit="s")
    display_df["outcome"] = display_df["resolved_up"].map(
        {1: "UP ✓", 0: "DOWN", None: "—"}
    )
    display_df["p_model"] = display_df["p_model"].round(4)
    display_df["market_mid"] = display_df["market_mid"].round(4)
    display_df["edge"] = display_df["edge"].round(4)
    display_df["pnl_net"] = display_df["pnl_net"].round(3)
    st.dataframe(
        display_df[["datetime", "condition_id", "direction", "p_model",
                    "market_mid", "edge", "bet_usdc", "outcome", "pnl_net"]],
        use_container_width=True,
    )

st.caption(f"DB: {SQLITE_DB_PATH} — Auto-refreshes every 10s")

# Auto-refresh
st.empty()
time.sleep(10)
st.rerun()
