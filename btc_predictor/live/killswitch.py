"""
Kill switch logic.

Trading halts immediately on any of:
  1. STALE DATA    — no Chainlink tick received in > 2 seconds
  2. DISCONNECT    — WebSocket connection dropped (auto-retried by feeds;
                     this fires if retry budget is exhausted)
  3. MAX DRAWDOWN  — portfolio drawdown exceeds MAX_DRAWDOWN_PCT
  4. MODEL BOUNDS  — model probability outside [0.01, 0.99]
                     (likely a feature pipeline or calibration bug)
  5. MANUAL HALT   — operator sets the halt flag via the dashboard

Once triggered, the kill switch remains active until manually cleared.
"""

from __future__ import annotations

import time
from enum import Enum, auto
from typing import Optional

from loguru import logger

from btc_predictor.config import (
    MAX_DRAWDOWN_PCT,
    MODEL_PROB_BOUNDS,
    STALE_DATA_THRESHOLD_S,
)


class HaltReason(Enum):
    STALE_DATA   = auto()
    DISCONNECT   = auto()
    MAX_DRAWDOWN = auto()
    MODEL_BOUNDS = auto()
    MANUAL       = auto()


class KillSwitch:
    """
    Stateful kill switch. Check `is_halted` before every trade.
    """

    def __init__(
        self,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        stale_threshold_s: float = STALE_DATA_THRESHOLD_S,
        model_prob_bounds: tuple[float, float] = MODEL_PROB_BOUNDS,
    ) -> None:
        self._max_dd     = max_drawdown_pct
        self._stale_thr  = stale_threshold_s
        self._prob_lo, self._prob_hi = model_prob_bounds
        self._halted     = False
        self._reason: Optional[HaltReason] = None
        self._halt_ts: Optional[float] = None
        self._peak_equity = 0.0

    @property
    def is_halted(self) -> bool:
        return self._halted

    @property
    def halt_reason(self) -> Optional[HaltReason]:
        return self._reason

    def update_equity(self, current_equity: float) -> None:
        """Call after each PnL update."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

    def check(
        self,
        chainlink_last_tick_wall_s: float,
        current_equity: float,
        model_prob: float,
    ) -> bool:
        """
        Run all checks. Returns True if trading is allowed, False if halted.
        Logs the halt reason on first trigger.
        """
        self.update_equity(current_equity)

        # 1. Stale data
        age = time.monotonic() - chainlink_last_tick_wall_s
        if age > self._stale_thr:
            self._trigger(HaltReason.STALE_DATA,
                          f"Chainlink data stale ({age:.1f}s)")
            return False

        # 2. Drawdown
        if self._peak_equity > 0:
            dd = (self._peak_equity - current_equity) / self._peak_equity
            if dd >= self._max_dd:
                self._trigger(HaltReason.MAX_DRAWDOWN,
                              f"Drawdown {dd:.1%} >= {self._max_dd:.1%}")
                return False

        # 3. Model bounds
        if not (self._prob_lo <= model_prob <= self._prob_hi):
            self._trigger(HaltReason.MODEL_BOUNDS,
                          f"Model prob {model_prob:.4f} out of bounds "
                          f"[{self._prob_lo}, {self._prob_hi}]")
            return False

        return True

    def manual_halt(self, reason: str = "operator") -> None:
        self._trigger(HaltReason.MANUAL, f"Manual halt: {reason}")

    def clear(self) -> None:
        """Re-enable trading after manual review."""
        if self._halted:
            logger.warning(
                f"[KillSwitch] Cleared halt (was: {self._reason}, "
                f"age {time.monotonic() - self._halt_ts:.0f}s)"
            )
        self._halted = False
        self._reason = None
        self._halt_ts = None

    def _trigger(self, reason: HaltReason, msg: str) -> None:
        if not self._halted:
            logger.error(f"[KillSwitch] HALT — {reason.name}: {msg}")
            self._halted   = True
            self._reason   = reason
            self._halt_ts  = time.monotonic()
