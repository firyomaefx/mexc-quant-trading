import sys

import asyncio
import numpy as np
import pandas as pd
import time
import signal
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime
from collections import deque

from config.crypto_config import CryptoConfig, PairConfig
from config.pairs import get_pair_config, list_enabled_symbols
from live.mexc_adapter import MEXCConnector
from live.mexc_hybrid import MEXCHybridConnector
from live.mexc_futures import MEXCFuturesConnector
from live.mexc_ws import MEXCWebSocket
from signals.generator import SignalGenerator
from signals.sentiment import SentimentAnalyzer
from signals.llm_sentiment import LLMSentimentAnalyzer
from signals.mtf_filter import MTFFilter
from signals.ml_ensemble import MLSignalEnhancer
from signals.rl_agent import RLAgent
from signals.pair_correlation import CorrelationFilter
from risk.circuit_breaker import CircuitBreaker
from risk.portfolio_risk import PortfolioRiskManager
from risk.futures_risk import FuturesRiskManager
from risk.exits_v2 import combined_exit
from risk.stops import atr_from_df
from config.settings import ThresholdConfig, WindowConfig, GoldConfig


class Scalper:
    def __init__(self, config: CryptoConfig, api_key: str = "", api_secret: str = "",
                 testnet: bool = False, use_hybrid: bool = True, proxy: str = None):
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        use_futures = config.futures.enabled
        if use_futures:
            self.mexc = MEXCFuturesConnector(api_key, api_secret, testnet)
        elif use_hybrid:
            self.mexc = MEXCHybridConnector(
                api_key=api_key, api_secret=api_secret, testnet=testnet,
                proxy=proxy, use_coingecko=True,
            )
        else:
            self.mexc = MEXCConnector(api_key, api_secret, testnet)

        market_type = "swap" if use_futures else "spot"
        self.ws = MEXCWebSocket(market_type=market_type)

        self.signal_gens: Dict[str, SignalGenerator] = {}
        self.sentiment = LLMSentimentAnalyzer(config.sentiment)
        self.mtf_filter = MTFFilter()
        self.ml_enhancer = MLSignalEnhancer(config.ml)
        self.rl_agent = RLAgent()
        self.correlation = CorrelationFilter()
        self.circuit_breaker = CircuitBreaker(
            initial_capital=config.scalping.initial_capital,
            max_daily_loss_pct=config.scalping.max_daily_loss_pct,
            max_drawdown_pct=config.scalping.max_drawdown_pct,
            max_consecutive_losses=config.scalping.max_consecutive_losses,
            cooldown_seconds=config.scalping.cooldown_seconds,
        )
        self.portfolio = PortfolioRiskManager(config)
        self.futures_risk = FuturesRiskManager(config.futures)
        self._highest_since_entry: Dict[str, float] = {}
        self._prev_zscore: Dict[str, float] = {}
        self._equity = config.scalping.initial_capital

        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}
        self._open_positions: Dict[str, Dict] = {}
        self._trade_history: deque = deque(maxlen=500)
        self._signal_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._symbols: List[str] = []
        self._bar_counters: Dict[str, int] = {}

    def connect(self, symbols: List[str] = None) -> bool:
        if symbols is None:
            symbols = list_enabled_symbols(self.config)
        self._symbols = symbols

        if not self.mexc.connect():
            print("Cannot connect to MEXC.")
            return False

        for sym in symbols:
            pair_cfg = get_pair_config(sym)
            self._setup_signal_generator(sym, pair_cfg)
            self._load_initial_data(sym)

        for sym in symbols:
            self.ws.subscribe_kline(sym, "Min1")
            self.ws.subscribe_ticker(sym)
        self.ws.on("candle", self._on_candle)
        self.ws.on("ticker", self._on_ticker)

        self.ws.start()
        print(f"Scalper initialized: {len(symbols)} pairs, "
              f"futures={self.config.futures.enabled}, "
              f"capital=${self.config.scalping.initial_capital:.0f}")

        if self.config.futures.enabled and self.config.futures.max_leverage > 1:
            for sym in symbols:
                self.mexc.set_leverage(sym, self.config.futures.max_leverage)
                self.mexc.set_margin_mode(sym, self.config.futures.margin_mode)

        return True

    def _setup_signal_generator(self, symbol: str, pair_cfg: PairConfig):
        threshold = ThresholdConfig(
            hurst_mean_revert=0.35,
            zscore_entry_long=pair_cfg.zscore_entry_long,
            zscore_entry_short=pair_cfg.zscore_entry_short,
            zscore_stop_long=pair_cfg.zscore_stop_long,
            zscore_stop_short=pair_cfg.zscore_stop_short,
            zscore_exit_target=0.0,
            velocity_epsilon=3.0,
            hmm_ranging_prob=0.0,
            time_stop_bars=pair_cfg.time_stop_bars,
        )
        win = WindowConfig(rolling_zscore=100, rolling_ma=20, hurst_max_lag=20)
        dummy = GoldConfig()
        dummy.threshold = threshold
        dummy.window = win
        self.signal_gens[symbol] = SignalGenerator(dummy)

    def _load_initial_data(self, symbol: str):
        try:
            df = self.mexc.fetch_rates(symbol, "1m", 500)
            self._df_cache[symbol] = df
            pair_cfg = get_pair_config(symbol)
            sig_gen = self.signal_gens[symbol]
            features = sig_gen.compute_and_generate(df)
            features["close"] = df.loc[features.index, "close"].values
            if "volume" in df.columns:
                features["volume"] = df.loc[features.index, "volume"].values
            self._features_cache[symbol] = features
            self._bar_counters[symbol] = len(df)
            print(f"  {symbol}: {len(df)} bars loaded")
        except Exception as e:
            print(f"  {symbol}: data load failed - {e}")
            self._df_cache[symbol] = pd.DataFrame()

    async def _on_candle(self, symbol: str, interval: str, candle: Dict):
        if symbol not in self._symbols:
            return
        await self._signal_queue.put(("candle", symbol, interval, candle))

    async def _on_ticker(self, symbol: str, ticker: Dict):
        if symbol not in self._symbols:
            return

    async def start(self, symbols: List[str] = None):
        if not self.connect(symbols):
            return

        self._running = True
        print(f"\n{'='*60}")
        print(f"  LIVE SCALPER RUNNING - {len(self._symbols)} pairs")
        print(f"  Capital: ${self.config.scalping.initial_capital:.0f}")
        print(f"  Risk/trade: {self.config.scalping.account_risk_pct*100:.1f}%")
        print(f"  Max positions: {self.config.scalping.max_concurrent_positions}")
        print(f"{'='*60}\n")

        loop = asyncio.get_event_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(s, lambda: asyncio.ensure_future(self.stop()))
            except NotImplementedError:
                pass

        try:
            while self._running:
                try:
                    event = await asyncio.wait_for(self._signal_queue.get(), timeout=10)
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    print(f"Event processing error: {e}")

        except asyncio.CancelledError:
            pass
        finally:
            self.cleanup()

    async def _process_event(self, event: Tuple):
        event_type = event[0]

        if event_type == "candle":
            _, symbol, interval, candle = event
            await self._process_candle(symbol, candle)

    async def _process_candle(self, symbol: str, candle: Dict):
        if not candle.get("closed", True):
            return

        df = self._df_cache.get(symbol)
        if df is None or len(df) < 100:
            await self._refresh_data(symbol)
            df = self._df_cache.get(symbol)

        new_row = pd.DataFrame([{
            "open": candle["open"],
            "high": candle["high"],
            "low": candle["low"],
            "close": candle["close"],
            "volume": candle["volume"],
        }], index=[pd.Timestamp(candle["time"], unit="ms")])

        df = pd.concat([df, new_row])
        if len(df) > 2000:
            df = df.iloc[-2000:]
        self._df_cache[symbol] = df

        sig_gen = self.signal_gens.get(symbol)
        if sig_gen is None:
            return

        features = sig_gen.compute_and_generate(df)
        if features.empty:
            return

        self._features_cache[symbol] = features
        latest = features.iloc[-1]
        signal_raw = int(latest.get("signal", 0))
        zscore = float(latest.get("zscore", 0))
        hurst = float(latest.get("hurst", 0.5))
        current_price = candle["close"]

        self.correlation.update(symbol, current_price)

        pos = self._open_positions.get(symbol)
        if pos is not None:
            await self._check_exit(symbol, zscore, current_price, pos, features)
            return

        if signal_raw == 0:
            return

        await self._check_entry(symbol, signal_raw, zscore, current_price, features, df)

    async def _check_exit(self, symbol: str, zscore: float, current_price: float,
                           pos: Dict, features: pd.DataFrame):
        pair_cfg = get_pair_config(symbol)

        highest_key = symbol
        if highest_key not in self._highest_since_entry:
            self._highest_since_entry[highest_key] = current_price if pos["direction"] == 1 else current_price
        else:
            if pos["direction"] == 1:
                self._highest_since_entry[highest_key] = max(self._highest_since_entry[highest_key], current_price)
            else:
                self._highest_since_entry[highest_key] = min(self._highest_since_entry[highest_key], current_price)

        prev_z = self._prev_zscore.get(symbol, 0.0)

        current_hurst = float(features.iloc[-1].get("hurst", 0.5)) if len(features) > 0 else 0.5
        entry_hurst = pos.get("entry_hurst", 0.5)

        atr_vals = atr_from_df(self._df_cache.get(symbol, pd.DataFrame()), period=14)
        current_atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else current_price * 0.005

        should_exit, reason, exit_z = combined_exit(
            current_zscore=zscore,
            entry_zscore=pos["entry_zscore"],
            bar_index=len(features),
            entry_bar=pos["entry_bar"],
            signal_direction=pos["direction"],
            max_bars=pair_cfg.time_stop_bars,
            zscore_stop_long=pair_cfg.zscore_stop_long,
            zscore_stop_short=pair_cfg.zscore_stop_short,
            current_price=current_price,
            highest_since_entry=self._highest_since_entry[highest_key],
            atr=current_atr,
            atr_trailing_mult=pair_cfg.atr_multiplier_sl if hasattr(pair_cfg, 'atr_multiplier_sl') else 1.5,
            prev_zscore=prev_z,
            zscore_velocity_threshold=0.3,
            current_hurst=current_hurst,
            entry_hurst=entry_hurst,
            hurst_exit_threshold=0.55,
        )

        self._prev_zscore[symbol] = zscore

        if not should_exit:
            if self.config.futures.enabled:
                liq_warn, liq_msg = self.futures_risk.monitor_position(
                    symbol, current_price, pos.get("liquidation_price", 0),
                    "long" if pos["direction"] == 1 else "short",
                    pos.get("leverage", 1)
                )
                if liq_warn:
                    print(f"  [{symbol}] {liq_msg}")
            return

        if self.config.futures.enabled:
            result = self.mexc.close_position(symbol, pos["amount"])
        else:
            result = self.mexc.close_position_spot(symbol, pos["amount"])
        if not result:
            print(f"  [{symbol}] Exit FAILED: {reason}")
            return

        pnl = (current_price - pos["entry_price"]) * pos["direction"] * pos["amount"]
        if self.config.futures.enabled:
            positions = self.mexc.get_positions(symbol)
            for fp in positions:
                if fp["symbol"] == symbol:
                    pnl = fp.get("unrealized_pnl", pnl)
                    break

        self._equity += pnl
        self.portfolio.update_equity(self._equity)
        self.ml_enhancer.update_trade_history(pnl, {}, pos["direction"])
        self.portfolio.update_pair_performance(symbol, list(self._trade_history))
        self.portfolio.release_position(symbol)
        self._highest_since_entry.pop(symbol, None)
        self._prev_zscore.pop(symbol, None)

        trade = {
            "symbol": symbol,
            "direction": "long" if pos["direction"] == 1 else "short",
            "entry_price": pos["entry_price"],
            "exit_price": current_price,
            "pnl": round(pnl, 4),
            "reason": reason,
            "entry_time": pos.get("entry_time", datetime.now()),
            "exit_time": datetime.now(),
        }
        self._trade_history.append(trade)

        del self._open_positions[symbol]
        print(f"  [{symbol}] EXIT {pos['direction_label']}: ${pnl:.4f} ({reason})")

    async def _check_entry(self, symbol: str, signal: int, zscore: float,
                           current_price: float, features: pd.DataFrame,
                           df: pd.DataFrame):
        pair_cfg = get_pair_config(symbol)

        active_positions = [
            {"symbol": s, "side": "buy" if p["direction"] == 1 else "sell"}
            for s, p in self._open_positions.items()
        ]

        max_conc = self.config.scalping.max_concurrent_positions
        if len(active_positions) >= max_conc:
            return

        sentiment_score = self.sentiment.get_sentiment(symbol)
        spread_pct = pair_cfg.typical_spread_pct

        bid, ask = 0, 0
        try:
            bid, ask = self.mexc.get_current_price(symbol)
            if bid > 0 and ask > 0:
                spread_pct = (ask - bid) / bid * 100
        except Exception:
            pass

        ml_conf = 1.0
        if self._features_cache.get(symbol) is not None:
            ml_features = MLSignalEnhancer.compute_ml_features(
                self._features_cache[symbol], spread_pct
            )
            ml_ok, ml_conf, ml_reason = self.ml_enhancer.should_trade(
                ml_features.iloc[-1].to_dict(), signal
            )
            if not ml_ok:
                return
        else:
            ml_conf = 1.0

        mtf_pass, mtf_conf, mtf_reason = self.mtf_filter.evaluate(
            df, None, None, signal
        )

        rl_conf = 1.0
        rl_ok = True
        if self.rl_agent._fitted and self._features_cache.get(symbol) is not None:
            try:
                obs = RLAgent.build_observation(self._features_cache[symbol])
                rl_ok, rl_conf, rl_reason = self.rl_agent.should_trade(obs, signal)
            except Exception:
                rl_ok = True
                rl_conf = 1.0
        else:
            rl_conf = 1.0

        if not rl_ok:
            return

        corr_pass, corr_reason = self.correlation.check_same_direction_block(
            symbol, signal, active_positions
        )

        atr_vals = atr_from_df(df, period=14)
        current_atr = atr_vals[-1] if not np.isnan(atr_vals[-1]) else current_price * 0.005
        atr_pct = current_atr / current_price if current_price > 0 else 0.01
        vol_ratio = atr_pct / 0.005

        allowed, breaker_reason = self.circuit_breaker.check_trade_allowed(
            symbol, zscore, spread_pct, vol_ratio
        )

        if not allowed:
            return
        if not mtf_pass:
            return
        if not corr_pass:
            return

        entry_price = current_price
        if signal == 1:
            entry_price = ask if ask > 0 else current_price
        else:
            entry_price = bid if bid > 0 else current_price

        pos_size = self.portfolio.calculate_position_size(
            symbol, self._equity, entry_price, current_atr,
            ml_conf=ml_conf, mtf_conf=mtf_conf,
            sentiment=sentiment_score, zscore_abs=abs(zscore),
            rl_conf=rl_conf,
        )

        pos_size = max(pair_cfg.min_qty, pos_size)
        notional = pos_size * entry_price
        if notional < pair_cfg.min_notional:
            pos_size = pair_cfg.min_notional / max(entry_price, 0.01)

        order_type = "BUY" if signal == 1 else "SELL"
        result = self.mexc.place_order(symbol, order_type, pos_size)

        if result is None:
            return

        pos_data = {
            "symbol": symbol,
            "direction": signal,
            "direction_label": "LONG" if signal == 1 else "SHORT",
            "entry_price": entry_price,
            "entry_zscore": zscore,
            "entry_bar": len(features),
            "entry_hurst": float(features.iloc[-1].get("hurst", 0.5)) if len(features) > 0 else 0.5,
            "amount": pos_size,
            "entry_time": datetime.now(),
            "leverage": self.config.futures.max_leverage if self.config.futures.enabled else 1,
            "liquidation_price": 0.0,
            "ml_conf": ml_conf,
            "mtf_conf": mtf_conf,
            "rl_conf": rl_conf,
            "sentiment": sentiment_score,
        }

        if self.config.futures.enabled:
            self.mexc.place_sl_tp(
                symbol, order_type, pos_size,
                entry_price * (1 - 0.015 * signal),
                entry_price * (1 + 0.02 * signal)
            )

        self._highest_since_entry[symbol] = entry_price
        self._prev_zscore[symbol] = zscore

        self._open_positions[symbol] = pos_data
        print(f"  [{symbol}] ENTRY {pos_data['direction_label']}: "
              f"${entry_price:.4f} x {pos_size:.4f}  "
              f"Z:{zscore:.2f}  S:{sentiment_score:+.2f}  "
              f"ML:{ml_conf:.2f}  RL:{rl_conf:.2f}")

    async def _refresh_data(self, symbol: str):
        try:
            df = self.mexc.fetch_rates(symbol, "1m", 500)
            if df is not None and len(df) > 0:
                self._df_cache[symbol] = df
                sig_gen = self.signal_gens.get(symbol)
                if sig_gen:
                    features = sig_gen.compute_and_generate(df)
                    features["close"] = df.loc[features.index, "close"].values
                    if "volume" in df.columns:
                        features["volume"] = df.loc[features.index, "volume"].values
                    self._features_cache[symbol] = features
        except Exception:
            pass

    async def stop(self):
        self._running = False
        self.cleanup()

    def cleanup(self):
        print("\nShutting down...")
        self.ws.close()
        self.mexc.disconnect()
        if self._open_positions:
            print(f"Open positions: {list(self._open_positions.keys())}")

        trades = list(self._trade_history)
        if trades:
            n = len(trades)
            wins = [t for t in trades if t["pnl"] > 0]
            wr = len(wins) / n if n > 0 else 0
            total_pnl = sum(t["pnl"] for t in trades)
            print(f"\n{'='*50}")
            print(f"  SCALPER SESSION SUMMARY")
            print(f"{'='*50}")
            print(f"  Trades: {n}  |  Win Rate: {wr:.1%}")
            print(f"  Total PnL: ${total_pnl:.2f}")
            print(f"{'='*50}")
        else:
            print("No trades executed.")

    def run(self):
        try:
            asyncio.run(self.start())
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            self.cleanup()
