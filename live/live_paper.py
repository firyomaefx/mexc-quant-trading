"""
Quant V2 - Live Paper Trading Engine
Continuously fetches MEXC data, generates signals, updates dashboard in real-time.
"""
import os
import sys
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from collections import deque
from typing import Dict, List

from config.crypto_config import CryptoConfig
from config.pairs import get_pair_config, list_enabled_symbols
from live.mexc_hybrid import MEXCHybridConnector
from signals.generator import SignalGenerator
from signals.llm_sentiment import LLMSentimentAnalyzer
from signals.mtf_filter import MTFFilter
from signals.ml_ensemble import MLSignalEnhancer
from signals.rl_agent import RLAgent
from signals.pair_correlation import CorrelationFilter
from risk.circuit_breaker import CircuitBreaker
from risk.portfolio_risk import PortfolioRiskManager
from risk.exits_v2 import combined_exit
from risk.stops import atr_from_df
from config.settings import ThresholdConfig, WindowConfig


class LivePaperEngine:
    def __init__(self, config: CryptoConfig, mexc: MEXCHybridConnector):
        self.config = config
        self.mexc = mexc
        self.pairs = list_enabled_symbols(config)
        self.running = False

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

        self._open_positions: Dict[str, Dict] = {}
        self._trade_history: deque = deque(maxlen=500)
        self._equity = config.scalping.initial_capital
        self._peak_equity = config.scalping.initial_capital
        self._daily_pnl = 0.0
        self._df_cache: Dict[str, pd.DataFrame] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}
        self._highest_since_entry: Dict[str, float] = {}
        self._prev_zscore: Dict[str, float] = {}

        self._lock = threading.Lock()
        self._last_bar: Dict[str, int] = {}

        for sym in self.pairs:
            pair_cfg = get_pair_config(sym)
            self._setup_signal_generator(sym, pair_cfg)

    def _setup_signal_generator(self, symbol: str, pair_config):
        threshold = ThresholdConfig(
            hurst_mean_revert=0.35,
            zscore_entry_long=pair_config.zscore_entry_long,
            zscore_entry_short=pair_config.zscore_entry_short,
            zscore_stop_long=pair_config.zscore_stop_long,
            zscore_stop_short=pair_config.zscore_stop_short,
            zscore_exit_target=0.0,
            velocity_epsilon=3.0,
            hmm_ranging_prob=0.0,
            time_stop_bars=pair_config.time_stop_bars,
        )
        win = WindowConfig(rolling_zscore=100, rolling_ma=20, hurst_max_lag=20)
        from config.settings import GoldConfig
        dummy = GoldConfig()
        dummy.threshold = threshold
        dummy.window = win
        self.signal_gens[symbol] = SignalGenerator(dummy)

    def load_initial_data(self):
        for sym in self.pairs:
            try:
                df = self.mexc.fetch_rates(sym, "1m", 500)
                if df is not None and len(df) > 0:
                    self._df_cache[sym] = df
                    sig_gen = self.signal_gens.get(sym)
                    if sig_gen:
                        features = sig_gen.compute_and_generate(df)
                        features["close"] = df.loc[features.index, "close"].values
                        if "volume" in df.columns:
                            features["volume"] = df.loc[features.index, "volume"].values
                        self._features_cache[sym] = features
                        self._last_bar[sym] = len(df)
                    print(f"  Loaded {len(df)} bars for {sym}")
                else:
                    print(f"  No data for {sym}")
            except Exception as e:
                print(f"  Error loading {sym}: {e}")

    def _fetch_new_bar(self, symbol: str) -> bool:
        try:
            df = self.mexc.fetch_rates(symbol, "1m", 5)
            if df is None or len(df) == 0:
                return False
            cached = self._df_cache.get(symbol)
            if cached is not None and len(cached) > 0:
                last_cached_time = cached.index[-1]
                new_rows = df[df.index > last_cached_time]
                if len(new_rows) > 0:
                    self._df_cache[symbol] = pd.concat([cached, new_rows]).iloc[-2000:]
                    self._update_features(symbol)
                    return True
            else:
                self._df_cache[symbol] = df.iloc[-500:]
                self._update_features(symbol)
                return True
        except Exception as e:
            return False

    def _update_features(self, symbol: str):
        df = self._df_cache.get(symbol)
        if df is None or len(df) < 100:
            return
        sig_gen = self.signal_gens.get(symbol)
        if sig_gen is None:
            return
        features = sig_gen.compute_and_generate(df)
        if features.empty:
            return
        features["close"] = df.loc[features.index, "close"].values
        if "volume" in df.columns:
            features["volume"] = df.loc[features.index, "volume"].values
        self._features_cache[symbol] = features

    def _process_signal(self, symbol: str):
        features = self._features_cache.get(symbol)
        df = self._df_cache.get(symbol)
        if features is None or df is None or len(features) < 2:
            return

        pair_cfg = get_pair_config(symbol)
        latest = features.iloc[-1]
        signal = int(latest.get("signal", 0))
        zscore = float(latest.get("zscore", 0))
        hurst = float(latest.get("hurst", 0.5))
        current_price = float(df["close"].iloc[-1])

        self.correlation.update(symbol, current_price)

        pos = self._open_positions.get(symbol)
        if pos is not None:
            self._check_exit(symbol, zscore, current_price, pos, features, df)
            return

        if signal == 0:
            return

        self._check_entry(symbol, signal, zscore, current_price, features, df)

    def _check_exit(self, symbol: str, zscore: float, current_price: float,
                     pos: Dict, features: pd.DataFrame, df: pd.DataFrame):
        pair_cfg = get_pair_config(symbol)
        pos_key = symbol

        if pos_key not in self._highest_since_entry:
            self._highest_since_entry[pos_key] = current_price if pos["direction"] == 1 else current_price
        else:
            if pos["direction"] == 1:
                self._highest_since_entry[pos_key] = max(self._highest_since_entry[pos_key], current_price)
            else:
                self._highest_since_entry[pos_key] = min(self._highest_since_entry[pos_key], current_price)

        prev_z = self._prev_zscore.get(symbol, 0.0)
        current_hurst = float(features.iloc[-1].get("hurst", 0.5)) if len(features) > 0 else 0.5
        entry_hurst = pos.get("entry_hurst", 0.5)

        atr_vals = atr_from_df(df, period=14)
        current_atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else current_price * 0.005

        should_exit, reason, exit_z = combined_exit(
            current_zscore=zscore, entry_zscore=pos["entry_zscore"],
            bar_index=len(features), entry_bar=pos["entry_bar"],
            signal_direction=pos["direction"], max_bars=pair_cfg.time_stop_bars,
            zscore_stop_long=pair_cfg.zscore_stop_long,
            zscore_stop_short=pair_cfg.zscore_stop_short,
            current_price=current_price,
            highest_since_entry=self._highest_since_entry[pos_key],
            atr=current_atr,
            atr_trailing_mult=pair_cfg.atr_multiplier_sl if hasattr(pair_cfg, 'atr_multiplier_sl') else 1.5,
            prev_zscore=prev_z, current_hurst=current_hurst,
            entry_hurst=entry_hurst, hurst_exit_threshold=0.55,
        )

        self._prev_zscore[symbol] = zscore

        if not should_exit:
            return

        spread_pct = pair_cfg.typical_spread_pct
        slippage = current_price * spread_pct / 100.0
        fill_exit = current_price - slippage * pos["direction"]

        pnl = (fill_exit - pos["entry_price"]) * pos["direction"] * pos["amount"]
        pnl += -fill_exit * pos["amount"] * self.config.backtest.commission_pct / 100.0 * 2

        self._equity += pnl
        self._daily_pnl += pnl

        trade = {
            "symbol": symbol,
            "direction": "long" if pos["direction"] == 1 else "short",
            "entry_price": pos["entry_price"],
            "exit_price": fill_exit,
            "pnl": round(pnl, 4),
            "reason": reason,
            "entry_bar": pos["entry_bar"],
            "exit_bar": len(features),
            "entry_time": pos.get("entry_time", datetime.now()),
            "exit_time": datetime.now(),
        }
        self._trade_history.append(trade)
        self.circuit_breaker.record_trade(symbol, pnl)
        self.portfolio.update_pair_performance(symbol, list(self._trade_history))
        self.portfolio.release_position(symbol)
        self._highest_since_entry.pop(symbol, None)
        self._prev_zscore.pop(symbol, None)

        del self._open_positions[symbol]
        eq_str = f"${self._equity:.2f}"
        dir_label = "LONG" if pos["direction"] == 1 else "SHORT"
        print(f"  [{datetime.now():%H:%M:%S}] {symbol} CLOSE {dir_label}: ${pnl:+.4f} ({reason})  Equity: {eq_str}")

    def _check_entry(self, symbol: str, signal: int, zscore: float,
                      current_price: float, features: pd.DataFrame, df: pd.DataFrame):
        pair_cfg = get_pair_config(symbol)

        active_positions = [
            {"symbol": s, "side": "buy" if p["direction"] == 1 else "sell"}
            for s, p in self._open_positions.items()
        ]
        if len(active_positions) >= self.config.scalping.max_concurrent_positions:
            return

        sentiment_score = self.sentiment.get_sentiment(symbol)
        spread_pct = pair_cfg.typical_spread_pct

        mtf_pass, mtf_conf, mtf_reason = self.mtf_filter.evaluate(df, None, None, signal)

        ml_conf = 1.0
        ml_ok = True
        if self._features_cache.get(symbol) is not None:
            try:
                ml_features = MLSignalEnhancer.compute_ml_features(
                    self._features_cache[symbol], spread_pct
                )
                ml_ok, ml_conf, ml_reason = self.ml_enhancer.should_trade(
                    ml_features.iloc[-1].to_dict(), signal
                )
                if not ml_ok:
                    return
            except Exception:
                ml_conf = 1.0

        rl_conf = 1.0
        rl_ok = True
        if self.rl_agent._fitted and symbol in self._features_cache:
            try:
                obs = RLAgent.build_observation(self._features_cache[symbol])
                rl_ok, rl_conf, rl_reason = self.rl_agent.should_trade(obs, signal)
                if not rl_ok:
                    return
            except Exception:
                rl_conf = 1.0

        corr_pass, _ = self.correlation.check_same_direction_block(symbol, signal, active_positions)
        if not corr_pass:
            return

        atr_vals = atr_from_df(df, period=14)
        current_atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else current_price * 0.005
        vol_ratio = (current_atr / current_price) / 0.005 if current_price > 0 else 1.0

        allowed, _ = self.circuit_breaker.check_trade_allowed(symbol, zscore, spread_pct, vol_ratio)
        if not allowed:
            return
        if not mtf_pass:
            return

        slippage = current_price * spread_pct / 100.0
        fill_price = current_price + slippage * signal

        position_size = self.portfolio.calculate_position_size(
            symbol, self._equity, fill_price, current_atr,
            ml_conf=ml_conf, mtf_conf=mtf_conf,
            sentiment=sentiment_score, zscore_abs=abs(zscore),
            rl_conf=rl_conf,
        )

        self._open_positions[symbol] = {
            "symbol": symbol,
            "direction": signal,
            "direction_label": "LONG" if signal == 1 else "SHORT",
            "entry_price": fill_price,
            "entry_zscore": zscore,
            "entry_bar": len(features),
            "entry_hurst": hurst,
            "amount": position_size,
            "entry_time": datetime.now(),
        }

        self._highest_since_entry[symbol] = fill_price
        self._prev_zscore[symbol] = zscore

        dir_label = "LONG" if signal == 1 else "SHORT"
        print(f"  [{datetime.now():%H:%M:%S}] {symbol} OPEN {dir_label}: "
              f"${fill_price:.4f} x {position_size:.4f}  Z:{zscore:.2f}  "
              f"S:{sentiment_score:+.2f}  ML:{ml_conf:.2f}  RL:{rl_conf:.2f}")

    def get_state(self) -> Dict:
        with self._lock:
            trades = list(self._trade_history)
            wins = [t for t in trades if t.get("pnl", 0) > 0] if trades else []
            wr = len(wins) / len(trades) * 100 if trades else 0
            pnls = [t.get("pnl", 0) for t in trades] if trades else []
            expectancy = np.mean(pnls) if pnls else 0
            peak = self._peak_equity
            dd = (peak - self._equity) / peak * 100 if peak > 0 else 0
            daily_loss = abs(self._daily_pnl) / self.config.scalping.initial_capital * 100

            try:
                cb = self.circuit_breaker.get_status()
            except Exception:
                cb = {}

            pair_data = {}
            for sym in self.pairs:
                df = self._df_cache.get(sym)
                feats = self._features_cache.get(sym)
                pdata = {"price": 0, "signal": 0, "zscore": 0, "hurst": 0.5, "ml_conf": 0}
                if df is not None and len(df) > 0:
                    pdata["price"] = float(df["close"].iloc[-1])
                if feats is not None and len(feats) > 0:
                    latest = feats.iloc[-1]
                    pdata["zscore"] = float(latest.get("zscore", 0))
                    pdata["hurst"] = float(latest.get("hurst", 0.5))
                    pdata["signal"] = int(latest.get("signal", 0))
                pair_data[sym] = pdata

            try:
                sentiment = self.sentiment.get_sentiment()
            except Exception:
                sentiment = 0

            return {
                "connected": self.mexc.connected,
                "equity": self._equity,
                "daily_pnl": self._daily_pnl,
                "win_rate": wr,
                "total_trades": len(trades),
                "expectancy": expectancy,
                "drawdown_pct": dd,
                "daily_loss_pct": daily_loss,
                "consecutive_losses": cb.get("consecutive_losses", 0),
                "breaker_status": cb.get("halt_reason", "OK"),
                "is_halted": cb.get("is_halted", False),
                "sentiment": sentiment,
                "pair_data": pair_data,
                "equity_curve": list(self._equity_curve)[-500:] if hasattr(self, '_equity_curve') else [self._equity],
                "trade_log": trades[-50:],
            }

    def run_loop(self, interval_seconds: int = 60):
        self.running = True
        self._equity_curve = [self._equity]

        print(f"\n{'='*60}")
        print(f"  LIVE PAPER TRADING STARTED")
        print(f"  Pairs: {self.pairs}")
        print(f"  Capital: ${self._equity:.2f}")
        print(f"  Interval: {interval_seconds}s")
        print(f"  Checking every {interval_seconds} seconds for new bars")
        print(f"{'='*60}\n")

        while self.running:
            try:
                for sym in self.pairs:
                    new_data = self._fetch_new_bar(sym)
                    if new_data:
                        self._process_signal(sym)

                with self._lock:
                    self._peak_equity = max(self._peak_equity, self._equity)
                    self._equity_curve.append(self._equity)
                    if len(self._equity_curve) > 500:
                        self._equity_curve = self._equity_curve[-500:]

            except Exception as e:
                print(f"  Loop error: {e}")

            time.sleep(interval_seconds)

    def stop(self):
        self.running = False
        trades = list(self._trade_history)
        if trades:
            wins = [t for t in trades if t.get("pnl", 0) > 0]
            total_pnl = sum(t.get("pnl", 0) for t in trades)
            print(f"\n{'='*50}")
            print(f"  SESSION SUMMARY")
            print(f"  Trades: {len(trades)}  Win Rate: {len(wins)/len(trades)*100:.1f}%")
            print(f"  P&L: ${total_pnl:.2f}  Final Equity: ${self._equity:.2f}")
            print(f"{'='*50}")
        else:
            print("\nNo trades executed.")