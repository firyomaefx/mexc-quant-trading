import os
import sys
import sys

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque

from config.crypto_config import CryptoConfig, PaperConfig, PairConfig
from config.pairs import get_pair_config, list_enabled_symbols
from signals.generator import SignalGenerator
from signals.sentiment import SentimentAnalyzer
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


class PaperTrader:
    def __init__(self, config: CryptoConfig, mexc_adapter=None):
        self.config = config
        self.paper_config: PaperConfig = config.paper
        self.mexc = mexc_adapter

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
        self._daily_start = config.scalping.initial_capital
        self._running = False
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._features_cache: Dict[str, pd.DataFrame] = {}
        self._highest_since_entry: Dict[str, float] = {}
        self._prev_zscore: Dict[str, float] = {}

    def load_historical_data(self, symbol: str, days: int = 30,
                             timeframe: str = "1m") -> pd.DataFrame:
        if self.mexc is None:
            return self._load_synthetic_data(symbol, days)

        try:
            limit = min(days * 1440, 1000)
            df = self.mexc.fetch_rates(symbol, timeframe=timeframe, limit=limit)
            self._data_cache[symbol] = df
            print(f"Loaded {len(df)} bars for {symbol}")
            return df
        except Exception as e:
            print(f"Historical data fetch failed for {symbol}: {e}")
            df = self._load_synthetic_data(symbol, days)
            return df

    def _load_synthetic_data(self, symbol: str, days: int) -> pd.DataFrame:
        from data.synthetic import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        n = min(days * 1440, 5000)
        df = gen.generate_regime_data(n_bars=n, n_ranging=int(n * 0.6), n_trending=int(n * 0.4), timeframe=1)
        self._data_cache[symbol] = df
        print(f"Generated {len(df)} synthetic bars for {symbol}")
        return df

    def setup_signal_generator(self, symbol: str, pair_config: PairConfig):
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

    def run_simulation(self, symbols: List[str] = None, max_bars: int = 500) -> Dict:
        if symbols is None:
            symbols = list_enabled_symbols(self.config)

        for sym in symbols:
            pair_cfg = get_pair_config(sym)
            self.setup_signal_generator(sym, pair_cfg)
            if sym not in self._data_cache:
                self.load_historical_data(sym)

        all_dfs = {sym: self._data_cache[sym] for sym in symbols if sym in self._data_cache}
        if not all_dfs:
            print("No data available for paper trading")
            return {"error": "no_data"}

        min_len = min(len(df) for df in all_dfs.values())
        n_bars = min(max_bars if max_bars > 0 else min_len, min_len)

        print(f"\n{'='*60}")
        print(f"  PAPER TRADING SIMULATION")
        print(f"  Pairs: {symbols}")
        print(f"  Bars: {n_bars}  |  Capital: ${self._equity:.0f}")
        print(f"{'='*60}\n")

        warmup = 100
        for i in range(warmup, n_bars):
            for sym in symbols:
                df = self._data_cache.get(sym)
                if df is None or i >= len(df):
                    continue
                self._process_bar(sym, df, i)

        report = self._generate_report()
        self._save_report(report)
        return report

    def _process_bar(self, symbol: str, df: pd.DataFrame, i: int):
        if i < 100:
            return

        pair_cfg = get_pair_config(symbol)
        window_df = df.iloc[:i + 1].copy()

        sig_gen = self.signal_gens.get(symbol)
        if sig_gen is None:
            self.setup_signal_generator(symbol, pair_cfg)
            sig_gen = self.signal_gens[symbol]

        features = sig_gen.compute_and_generate(window_df)
        if features.empty:
            return

        self._features_cache[symbol] = features

        latest = features.iloc[-1]
        signal = int(latest.get("signal", 0))
        zscore = float(latest.get("zscore", 0))
        hurst = float(latest.get("hurst", 0.5))
        current_price = float(window_df["close"].iloc[-1])

        self.correlation.update(symbol, current_price)

        pos = self._open_positions.get(symbol)
        if pos is not None:
            pos_key = symbol
            if pos_key not in self._highest_since_entry:
                self._highest_since_entry[pos_key] = current_price if pos["direction"] == 1 else current_price
            else:
                if pos["direction"] == 1:
                    self._highest_since_entry[pos_key] = max(self._highest_since_entry[pos_key], current_price)
                else:
                    self._highest_since_entry[pos_key] = min(self._highest_since_entry[pos_key], current_price)

            prev_z = self._prev_zscore.get(symbol, 0.0)
            current_hurst = hurst
            entry_hurst = pos.get("entry_hurst", 0.5)

            atr_vals = atr_from_df(window_df, period=14)
            current_atr = atr_vals[-1] if len(atr_vals) > 0 and not np.isnan(atr_vals[-1]) else current_price * 0.005

            should_exit, reason, exit_z = combined_exit(
                current_zscore=zscore,
                entry_zscore=pos["entry_zscore"],
                bar_index=i,
                entry_bar=pos["entry_bar"],
                signal_direction=pos["direction"],
                max_bars=pair_cfg.time_stop_bars,
                zscore_stop_long=pair_cfg.zscore_stop_long,
                zscore_stop_short=pair_cfg.zscore_stop_short,
                current_price=current_price,
                highest_since_entry=self._highest_since_entry[pos_key],
                atr=current_atr,
                atr_trailing_mult=pair_cfg.atr_multiplier_sl if hasattr(pair_cfg, 'atr_multiplier_sl') else 1.5,
                prev_zscore=prev_z,
                zscore_velocity_threshold=0.3,
                current_hurst=current_hurst,
                entry_hurst=entry_hurst,
                hurst_exit_threshold=0.55,
            )

            self._prev_zscore[symbol] = zscore

            if should_exit:
                exit_price = current_price
                slippage = exit_price * self.paper_config.slippage_pct / 100.0
                fill_exit = exit_price - slippage * pos["direction"]
                time.sleep(self.paper_config.latency_ms / 1000.0)

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
                    "exit_bar": i,
                    "entry_time": pos.get("entry_time", datetime.now()),
                    "exit_time": datetime.now(),
                    "paper": True,
                }
                self._trade_history.append(trade)
                self.circuit_breaker.record_trade(symbol, pnl)
                self.portfolio.update_pair_performance(symbol, list(self._trade_history))
                self.portfolio.release_position(symbol)
                self._highest_since_entry.pop(symbol, None)
                self._prev_zscore.pop(symbol, None)

                del self._open_positions[symbol]
                print(f"  [{symbol}] CLOSE {pos['direction_label']}: ${pnl:.4f} ({reason})  Equity: ${self._equity:.2f}")
            return

        if signal == 0:
            return

        sentiment_score = 0.0
        if self.mexc:
            sentiment_score = self.sentiment.get_sentiment(symbol)

        spread_pct = pair_cfg.typical_spread_pct

        mtf_pass, mtf_conf, mtf_reason = self.mtf_filter.evaluate(
            window_df, None, None, signal
        )

        ml_conf = 1.0
        if mtf_pass:
            features_with_price = features.copy()
            features_with_price["close"] = window_df.loc[features.index, "close"].values
            if "volume" in window_df.columns:
                features_with_price["volume"] = window_df.loc[features.index, "volume"].values
            ml_features_df = MLSignalEnhancer.compute_ml_features(features_with_price, spread_pct)
            ml_ok, ml_conf, ml_reason = self.ml_enhancer.should_trade(
                ml_features_df.iloc[-1].to_dict(), signal
            )
            if not ml_ok:
                return

        rl_conf = 1.0
        rl_ok = True
        if self.rl_agent._fitted and symbol in self._features_cache:
            try:
                obs = RLAgent.build_observation(self._features_cache[symbol])
                rl_ok, rl_conf, rl_reason = self.rl_agent.should_trade(obs, signal)
            except Exception:
                rl_ok = True
                rl_conf = 1.0

        if not rl_ok:
            return

        active_positions = [
            {"symbol": s, "side": "buy" if p["direction"] == 1 else "sell"}
            for s, p in self._open_positions.items()
        ]
        corr_pass, corr_reason = self.correlation.check_same_direction_block(
            symbol, signal, active_positions
        )

        allowed, breaker_reason = self.circuit_breaker.check_trade_allowed(
            symbol, zscore, spread_pct, 1.0, self._equity
        )

        if not allowed:
            return
        if not mtf_pass:
            return
        if not corr_pass:
            return

        entry_price = current_price
        slippage = entry_price * self.paper_config.slippage_pct / 100.0
        fill_price = entry_price + slippage * signal
        time.sleep(self.paper_config.latency_ms / 1000.0)

        atr_vals = atr_from_df(window_df, period=14)
        current_atr = atr_vals[-1] if not np.isnan(atr_vals[-1]) else entry_price * 0.005
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
            "entry_bar": i,
            "entry_hurst": hurst,
            "amount": position_size,
            "entry_time": datetime.now(),
        }

        self._highest_since_entry[symbol] = fill_price
        self._prev_zscore[symbol] = zscore

        print(f"  [{symbol}] OPEN {self._open_positions[symbol]['direction_label']}: "
              f"${fill_price:.4f} x {position_size:.4f}  Z:{zscore:.2f}  "
              f"S:{sentiment_score:+.2f}  ML:{ml_conf:.2f}  RL:{rl_conf:.2f}")

    def _generate_report(self) -> Dict:
        trades = list(self._trade_history)
        n = len(trades)
        if n == 0:
            return {"status": "no_trades", "message": "No trades executed in simulation"}

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        wr = len(wins) / n if n > 0 else 0
        total_pnl = sum(t["pnl"] for t in trades)
        avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_l = np.mean([t["pnl"] for t in losses]) if losses else 0
        expectancy = wr * avg_w - (1 - wr) * abs(avg_l)

        ready = (
            n >= self.paper_config.min_trades_for_live
            and wr >= self.paper_config.min_win_rate_for_live
            and expectancy >= self.paper_config.min_expectancy_for_live
        )

        return {
            "total_trades": n,
            "win_rate": round(wr * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_win": round(avg_w, 4),
            "avg_loss": round(avg_l, 4),
            "expectancy": round(expectancy, 4),
            "final_equity": round(self._equity, 2),
            "return_pct": round((self._equity / self.config.scalping.initial_capital - 1) * 100, 2),
            "ready_for_live": ready,
            "message": "READY FOR LIVE" if ready else f"Need {self.paper_config.min_trades_for_live - n} more trades",
            "trades": [
                {"symbol": t["symbol"], "pnl": t["pnl"], "reason": t["reason"]}
                for t in trades[-10:]
            ],
        }

    def _save_report(self, report: Dict):
        os.makedirs(self.paper_config.report_dir, exist_ok=True)
        filename = f"{self.paper_config.report_dir}/paper_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
        trades = list(self._trade_history)
        if trades:
            df = pd.DataFrame(trades)
            df.to_csv(filename, index=False)
            print(f"\nPaper report saved: {filename}")

    def print_report(self, report: Dict):
        print(f"\n{'='*60}")
        print(f"  PAPER TRADING REPORT")
        print(f"{'='*60}")
        print(f"  Trades: {report.get('total_trades', 0)}")
        print(f"  Win Rate: {report.get('win_rate', 0)}%")
        print(f"  Total PnL: ${report.get('total_pnl', 0):.2f}")
        print(f"  Avg Win: ${report.get('avg_win', 0):.4f}")
        print(f"  Avg Loss: ${report.get('avg_loss', 0):.4f}")
        print(f"  Expectancy: ${report.get('expectancy', 0):.4f}")
        print(f"  Final Equity: ${report.get('final_equity', 0):.2f}")
        print(f"  Return: {report.get('return_pct', 0):.2f}%")
        print(f"  Status: {report.get('message', 'N/A')}")
        print(f"{'='*60}\n")