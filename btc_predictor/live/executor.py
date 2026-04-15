"""
Live execution loop.

Runs every second:
  1. Compute features from live feeds
  2. Get model probability p
  3. Get Polymarket market mid m
  4. Compute edge = p - m (for UP) or (1-p) - (1-m) (for DOWN)
  5. Check kill switches
  6. Size position via fractional Kelly
  7. Submit limit order (or market in dry-run)
  8. Log prediction, fill, and PnL to SQLite

In --dry-run mode, orders are simulated and printed but not submitted.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from loguru import logger

from btc_predictor.config import (
    ENTRY_CUTOFF_S,
    KELLY_FRACTION,
    MARKET_WINDOW_SECONDS,
    MAX_KELLY_BET_USDC,
    MIN_EDGE,
    SAFETY_MARGIN,
    LiveConfig,
)
from btc_predictor.data.bar_builder import MultiExchangeBarStore
from btc_predictor.data.binance_feed import BinanceFeed
from btc_predictor.data.bybit_feed import BybitFeed
from btc_predictor.data.chainlink_feed import ChainlinkFeed
from btc_predictor.data.coinbase_feed import CoinbaseFeed
from btc_predictor.data.cme_basis import CMEBasisCollector
from btc_predictor.data.funding_rates import FundingRateCollector
from btc_predictor.data.polymarket_feed import PolymarketFeed
from btc_predictor.features.pipeline import FeaturePipeline
from btc_predictor.live.killswitch import KillSwitch
from btc_predictor.live.sizing import expected_value_per_dollar, fractional_kelly
from btc_predictor.models.ensemble import EnsemblePredictor
from btc_predictor.monitoring.logger import TradeLogger
from btc_predictor.backtest.fills import break_even_edge


class LiveExecutor:
    """
    Main live trading loop.

    Usage:
        cfg = LiveConfig(dry_run=True)
        executor = LiveExecutor(cfg)
        asyncio.run(executor.run())
    """

    def __init__(self, cfg: LiveConfig) -> None:
        self._cfg = cfg
        self._bankroll: float = 0.0   # loaded from DB or set to initial
        self._equity:   float = 0.0
        self._kill      = KillSwitch(
            max_drawdown_pct=cfg.max_drawdown_pct,
            stale_threshold_s=cfg.stale_data_threshold_s,
        )
        self._model: Optional[EnsemblePredictor] = None
        self._logger: Optional[TradeLogger] = None
        self._entered_this_market: bool = False
        self._current_market_open_ts: int = 0

    async def run(self) -> None:
        """Start all feeds, load model, then enter the main loop."""
        logger.info(
            f"[LiveExecutor] Starting — dry_run={self._cfg.dry_run}, "
            f"kelly={self._cfg.kelly_fraction:.0%}, "
            f"max_bet=${self._cfg.max_bet_usdc:.0f}"
        )

        # --- Infrastructure ---
        store    = MultiExchangeBarStore()
        chainlink = ChainlinkFeed()
        polymarket = PolymarketFeed()
        funding   = FundingRateCollector()
        cme       = CMEBasisCollector()
        binance   = BinanceFeed(store)
        coinbase  = CoinbaseFeed(store)
        bybit     = BybitFeed(store)

        self._logger = TradeLogger()
        self._logger.init_db()

        pipeline = FeaturePipeline(
            store=store,
            chainlink=chainlink,
            polymarket=polymarket,
            funding=funding,
            cme=cme,
        )

        # --- Load model ---
        self._model = EnsemblePredictor()
        try:
            self._model.load()
            logger.info("[LiveExecutor] Model loaded")
        except Exception as e:
            logger.error(f"[LiveExecutor] Failed to load model: {e}")
            raise

        self._bankroll = self._logger.get_current_equity(initial=10_000.0)
        self._equity   = self._bankroll
        self._kill.update_equity(self._equity)

        # --- Start feeds ---
        await asyncio.gather(
            chainlink.start(),
            polymarket.start(),
            funding.start(),
            cme.start(),
            binance.start(),
            coinbase.start(),
            bybit.start(),
        )

        logger.info("[LiveExecutor] All feeds started — entering main loop")

        try:
            await self._main_loop(pipeline, chainlink, polymarket)
        except asyncio.CancelledError:
            logger.info("[LiveExecutor] Cancelled")
        finally:
            await asyncio.gather(
                chainlink.stop(),
                polymarket.stop(),
                funding.stop(),
                cme.stop(),
                binance.stop(),
                coinbase.stop(),
                bybit.stop(),
                return_exceptions=True,
            )
            logger.info("[LiveExecutor] All feeds stopped")

    async def _main_loop(
        self,
        pipeline: FeaturePipeline,
        chainlink: ChainlinkFeed,
        polymarket: PolymarketFeed,
    ) -> None:
        """1-Hz decision loop."""
        while True:
            now_s = int(time.time())
            await asyncio.sleep(1.0 - (time.time() % 1.0))  # align to wall-clock second

            # Detect new market window
            market_open_ts = now_s - (now_s % MARKET_WINDOW_SECONDS)
            if market_open_ts != self._current_market_open_ts:
                self._current_market_open_ts = market_open_ts
                self._entered_this_market = False
                logger.info(f"[LiveExecutor] New market window: open_ts={market_open_ts}")

            # Check entry cutoff
            time_remaining = (market_open_ts + MARKET_WINDOW_SECONDS) - now_s
            if time_remaining <= self._cfg.entry_cutoff_s:
                continue   # too close to resolution

            # One entry per market
            if self._entered_this_market:
                continue

            # Compute features
            try:
                fv = await pipeline.compute(now_s)
            except Exception as e:
                logger.warning(f"[LiveExecutor] Feature compute error: {e}")
                continue

            # Model inference
            try:
                p = self._model.predict(fv.features, fv.sequence)
            except Exception as e:
                logger.warning(f"[LiveExecutor] Model inference error: {e}")
                continue

            # Kill switch checks
            cl_last_tick = (chainlink._last_tick_ts
                            if hasattr(chainlink, "_last_tick_ts") else 0.0)
            if not self._kill.check(
                chainlink_last_tick_wall_s=cl_last_tick,
                current_equity=self._equity,
                model_prob=p,
            ):
                logger.warning(
                    f"[LiveExecutor] Kill switch active: {self._kill.halt_reason}"
                )
                continue

            # Current market
            market = polymarket.current_market
            if market is None:
                continue

            market_mid = market.mid
            if market_mid <= 0.01 or market_mid >= 0.99:
                continue

            # Compute edge
            edge_up   = p - market_mid
            edge_down = (1.0 - p) - (1.0 - market_mid)
            be        = break_even_edge(market_mid)
            min_edge  = (be - market_mid) + self._cfg.safety_margin + self._cfg.min_edge

            direction = None
            edge = 0.0
            trade_prob = p

            if edge_up >= min_edge:
                direction  = "UP"
                edge       = edge_up
                trade_prob = p
            elif edge_down >= min_edge:
                direction  = "DOWN"
                edge       = edge_down
                trade_prob = 1.0 - p

            if direction is None:
                if self._cfg.verbose:
                    logger.debug(
                        f"[LiveExecutor] No edge: p={p:.3f} m={market_mid:.3f} "
                        f"edge_up={edge_up:.3f} min={min_edge:.3f}"
                    )
                continue

            # Kelly sizing
            effective_price = market_mid if direction == "UP" else (1.0 - market_mid)
            bet_usdc = fractional_kelly(
                p_model=trade_prob,
                market_price=effective_price,
                bankroll=self._equity,
                kelly_fraction=self._cfg.kelly_fraction,
                max_bet=self._cfg.max_bet_usdc,
            )

            if bet_usdc < 1.0:
                continue

            ev = expected_value_per_dollar(trade_prob, effective_price)

            logger.info(
                f"[LiveExecutor] SIGNAL {direction} "
                f"p={p:.4f} m={market_mid:.4f} "
                f"edge={edge:.4f} ev={ev:.4f} "
                f"bet=${bet_usdc:.2f} "
                f"{'[DRY-RUN]' if self._cfg.dry_run else ''}"
            )

            # Log prediction
            self._logger.log_prediction(
                ts_s=now_s,
                condition_id=market.condition_id,
                p_model=p,
                market_mid=market_mid,
                edge=edge,
                direction=direction,
                bet_usdc=bet_usdc,
            )

            # Execute (or simulate)
            if not self._cfg.dry_run:
                await self._submit_order(market, direction, bet_usdc)
            else:
                logger.info(
                    f"[DRY-RUN] Would place ${bet_usdc:.2f} {direction} "
                    f"on {market.condition_id}"
                )

            self._entered_this_market = True

    async def _submit_order(self, market, direction: str, bet_usdc: float) -> None:
        """
        Submit a real order to Polymarket CLOB.
        Requires L1/L2 authentication (API key + private key).
        See: https://docs.polymarket.com/developers/CLOB/clients/methods-overview

        Not implemented in skeleton — requires wallet credentials.
        Raises NotImplementedError until auth is wired up.
        """
        raise NotImplementedError(
            "Live order submission requires Polymarket API credentials. "
            "Set POLYMARKET_PRIVATE_KEY and POLYMARKET_API_KEY env vars, "
            "then implement this method using py-clob-client."
        )
