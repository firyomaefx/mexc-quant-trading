import sys

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time

from signals.generator import SignalGenerator
from stats.hurst import rolling_hurst
from stats.zscore import rolling_zscore
from stats.velocity import ma_velocity, velocity_approaching_zero
from risk.exits import combined_exit, apply_exits_to_df
from config.crypto_config import CryptoConfig, PairConfig, BacktestConfig
from config.pairs import get_pair_config, list_enabled_symbols
from config.settings import ThresholdConfig, WindowConfig
def _compute_metrics(equity: float, capital: float, trades: list, symbol: str) -> dict:
    if not trades:
        return None
    n = len(trades)
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    wr = len(wins) / n if n > 0 else 0
    pnls = [t.get("pnl", 0) for t in trades]
    total_pnl = sum(pnls)
    avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.get("pnl", 0) for t in losses])) if losses else 0
    ret_pct = (equity / capital - 1) * 100
    std_pnl = np.std(pnls) if len(pnls) > 1 else 0.01
    if avg_loss > 0 and len(pnls) > 1:
        sharpe = (np.mean(pnls) / max(std_pnl, 1e-10)) * np.sqrt(len(pnls))
    else:
        sharpe = 0.0
    return {
        "symbol": symbol,
        "return_pct": round(ret_pct, 2),
        "sharpe": round(sharpe, 2),
        "win_rate": wr,
        "total_trades": n,
        "total_pnl": round(total_pnl, 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "profit_factor": round(abs(sum(t.get("pnl", 0) for t in wins) / sum(t.get("pnl", 0) for t in losses)), 2) if losses and sum(t.get("pnl", 0) for t in losses) != 0 else 0,
    }


class WalkForwardBacktester:
    def __init__(self, config: CryptoConfig):
        self.config = config
        self.bt_config: BacktestConfig = config.backtest
        self._results: Dict[str, Dict] = {}

    def run_pair(self, symbol: str, df: pd.DataFrame,
                 train_months: int = 3, test_months: int = 1) -> Dict:
        pair_config = get_pair_config(symbol)

        if df is None or len(df) < 500:
            return {"error": f"Insufficient data for {symbol}: {len(df) if df is not None else 0} bars"}

        tf_minutes = self._infer_timeframe(df)
        bars_per_day = 1440 // tf_minutes if tf_minutes > 0 else 288
        train_bars = train_months * 30 * bars_per_day
        test_bars = test_months * 30 * bars_per_day

        windows = self._generate_windows(len(df), train_bars, test_bars)
        if not windows:
            return {"error": f"No walk-forward windows for {symbol}"}

        all_metrics = []
        all_trades = []

        threshold = ThresholdConfig(
            hurst_mean_revert=0.35,
            zscore_entry_long=pair_config.zscore_entry_long,
            zscore_entry_short=pair_config.zscore_entry_short,
            zscore_stop_long=pair_config.zscore_stop_long,
            zscore_stop_short=pair_config.zscore_stop_short,
            zscore_exit_target=0.0,
            velocity_epsilon=3.0,
            time_stop_bars=pair_config.time_stop_bars,
        )
        window_config = WindowConfig(rolling_zscore=100, rolling_ma=20, hurst_max_lag=20)

        for train_start, train_end, test_start, test_end in windows:
            if test_end - test_start < 50:
                continue

            test_df = df.iloc[test_start:test_end].copy()
            metrics, trades = self._run_window(
                symbol, test_df, threshold, window_config, pair_config
            )
            if metrics:
                all_metrics.append(metrics)
            if trades:
                all_trades.extend(trades)

        if not all_metrics:
            return {"error": f"No valid windows for {symbol}"}

        summary = self._summarize_metrics(all_metrics, all_trades, symbol)
        self._results[symbol] = summary
        return summary

    def _generate_windows(self, total_bars: int, train_bars: int,
                          test_bars: int) -> List[Tuple[int, int, int, int]]:
        windows = []
        start = train_bars
        while start + test_bars <= total_bars:
            win = (start - train_bars, start, start, start + test_bars)
            windows.append(win)
            start += test_bars
        return windows

    def _run_window(self, symbol: str, df: pd.DataFrame,
                    threshold: ThresholdConfig, window_config: WindowConfig,
                    pair_config: PairConfig) -> Tuple[Optional[Dict], List[Dict]]:
        if len(df) < 100:
            return None, []

        close = df["close"].values.astype(np.float64)
        n = len(close)

        try:
            zs = rolling_zscore(close, window=window_config.rolling_zscore)
            h_vals = rolling_hurst(close, window=window_config.rolling_zscore,
                                   max_lag=window_config.hurst_max_lag)
            velocity = ma_velocity(close, ma_period=window_config.rolling_ma)
            vel_zero = velocity_approaching_zero(velocity, epsilon=threshold.velocity_epsilon)
        except Exception:
            return None, []

        features = pd.DataFrame(index=df.index[-n:])
        features["zscore"] = zs["zscore"].values[-n:]
        features["hurst"] = h_vals[-n:]
        features["velocity"] = velocity[-n:]
        features["velocity_zero"] = vel_zero.astype(int)[-n:]
        features["close"] = close[-n:]

        is_mean_revert = features["hurst"] < threshold.hurst_mean_revert
        is_oversold = features["zscore"] < threshold.zscore_entry_long
        is_overbought = features["zscore"] > threshold.zscore_entry_short
        velocity_flat = features["velocity_zero"] == 1

        features["signal"] = 0
        features.loc[is_mean_revert & is_oversold & velocity_flat, "signal"] = 1
        features.loc[is_mean_revert & is_overbought & velocity_flat, "signal"] = -1

        signals_df = apply_exits_to_df(features, max_bars=threshold.time_stop_bars,
                                       zscore_stop_long=threshold.zscore_stop_long,
                                       zscore_stop_short=threshold.zscore_stop_short)

        entries = (signals_df["signal"] != 0).values
        exits = (signals_df["exit_signal"] != 0).values

        capital = self.bt_config.initial_capital
        balance = capital
        position = 0.0
        entry_price = 0.0
        entry_bar = 0
        direction = 0
        trades = []

        for i in range(len(signals_df)):
            price = signals_df["close"].iloc[i]
            if price <= 0:
                continue

            if direction == 0 and entries[i]:
                signal = int(signals_df["signal"].iloc[i])
                direction = signal
                entry_price = price
                entry_bar = i

                spread_cost = price * pair_config.typical_spread_pct / 100.0
                slippage = price * self.bt_config.slippage_pct / 100.0
                fill_price = price + (spread_cost + slippage) * signal

                min_notional = pair_config.min_notional
                position_value = min(capital * 0.15, balance * 0.5)
                position_value = max(min_notional, position_value)
                position = position_value / fill_price
                position = max(pair_config.min_qty, round(position / pair_config.qty_step) * pair_config.qty_step)
                position = round(position, 8)

                balance -= position_value * self.bt_config.commission_pct / 100.0

            elif direction != 0 and exits[i]:
                exit_price = price
                spread_cost = exit_price * pair_config.typical_spread_pct / 100.0
                slippage = exit_price * self.bt_config.slippage_pct / 100.0
                fill_exit = exit_price - (spread_cost + slippage) * direction

                pnl = (fill_exit - entry_price) * direction * position
                balance += pnl
                balance -= position * fill_exit * self.bt_config.commission_pct / 100.0

                trades.append({
                    "symbol": symbol,
                    "direction": "long" if direction == 1 else "short",
                    "entry_price": entry_price,
                    "exit_price": fill_exit,
                    "pnl": round(pnl, 4),
                    "pnl_pct": round(pnl / capital * 100, 4),
                    "bars_held": i - entry_bar,
                    "exit_reason": str(signals_df["exit_reason"].iloc[i]) if "exit_reason" in signals_df.columns else "",
                })

                position = 0.0
                direction = 0
                entry_price = 0.0

        equity = balance
        if trades:
            return _compute_metrics(equity, capital, trades, symbol), trades
        return None, []

    def run_monte_carlo(self, symbol: str, trades: List[Dict],
                        n_simulations: int = 1000) -> Dict:
        if not trades:
            return {"max_drawdown_pct": 0, "cvar_95": 0, "ruin_prob": 0}

        pnls = np.array([t["pnl"] for t in trades])
        n_trades = len(pnls)
        capital = self.bt_config.initial_capital

        drawdowns = []
        final_equities = []
        ruins = 0

        for _ in range(n_simulations):
            shuffled = np.random.choice(pnls, size=n_trades, replace=True)
            equity_curve = capital + np.cumsum(shuffled)
            peak = np.maximum.accumulate(equity_curve)
            dd = (peak - equity_curve) / np.where(peak > 0, peak, 1)
            max_dd = np.max(dd)
            drawdowns.append(max_dd)
            final_equities.append(equity_curve[-1])
            if equity_curve[-1] < capital * 0.5:
                ruins += 1

        drawdowns = np.array(drawdowns)
        final_equities = np.array(final_equities)

        return {
            "mean_max_dd_pct": round(float(np.mean(drawdowns)) * 100, 2),
            "worst_dd_pct": round(float(np.max(drawdowns)) * 100, 2),
            "cvar_95_dd_pct": round(float(np.percentile(drawdowns, 95)) * 100, 2),
            "mean_final_equity": round(float(np.mean(final_equities)), 2),
            "p10_equity": round(float(np.percentile(final_equities, 10)), 2),
            "p90_equity": round(float(np.percentile(final_equities, 90)), 2),
            "ruin_probability": round(ruins / n_simulations * 100, 2),
            "n_simulations": n_simulations,
        }

    def _summarize_metrics(self, all_metrics: List[Dict], all_trades: List[Dict],
                           symbol: str) -> Dict:
        if not all_metrics:
            return {"symbol": symbol, "error": "no_valid_windows"}

        n_windows = len(all_metrics)
        sharpe_vals = [m.get("sharpe", 0) for m in all_metrics if m.get("sharpe") is not None]
        return_vals = [m.get("return_pct", 0) for m in all_metrics]
        wr_vals = [m.get("win_rate", 0) for m in all_metrics]

        mc_results = {}
        if all_trades:
            mc_results = self.run_monte_carlo(symbol, all_trades, self.bt_config.monte_carlo_runs)

        summary = {
            "symbol": symbol,
            "trades": all_trades,
            "n_windows": n_windows,
            "total_trades": len(all_trades),
            "mean_sharpe": round(float(np.mean(sharpe_vals)), 2) if sharpe_vals else 0,
            "std_sharpe": round(float(np.std(sharpe_vals)), 2) if len(sharpe_vals) > 1 else 0,
            "mean_return_pct": round(float(np.mean(return_vals)), 2) if return_vals else 0,
            "mean_win_rate": round(float(np.mean(wr_vals)) * 100, 1) if wr_vals else 0,
            "total_pnl": round(sum(t["pnl"] for t in all_trades), 2),
            "monte_carlo": mc_results,
        }
        return summary

    def run_all_pairs(self, data: Dict[str, pd.DataFrame]) -> Dict:
        results = {}
        for symbol, df in data.items():
            try:
                res = self.run_pair(symbol, df,
                                    self.bt_config.train_months,
                                    self.bt_config.test_months)
                results[symbol] = res
                if "error" not in res:
                    print(f"  {symbol}: {res.get('mean_sharpe', 0):.2f} Sharpe, "
                          f"{res.get('mean_return_pct', 0):.1f}% ret, "
                          f"{res.get('total_trades', 0)} trades")
            except Exception as e:
                results[symbol] = {"error": str(e)}
                print(f"  {symbol}: ERROR - {e}")
        return results

    def _infer_timeframe(self, df: pd.DataFrame) -> int:
        if len(df) < 2:
            return 1
        diff = (df.index[1] - df.index[0]).total_seconds() / 60
        if 0.5 < diff < 2:
            return 1
        elif 2 <= diff < 4:
            return 3
        elif 4 <= diff < 10:
            return 5
        elif 10 <= diff < 20:
            return 15
        else:
            return 60

    def get_best_pair(self) -> Optional[str]:
        if not self._results:
            return None
        best = None
        best_score = -999
        for sym, res in self._results.items():
            score = res.get("mean_sharpe", 0)
            if score > best_score:
                best_score = score
                best = sym
        return best

    def print_summary(self):
        print("\n" + "=" * 70)
        print("  WALK-FORWARD BACKTEST SUMMARY")
        print("=" * 70)
        for sym, res in self._results.items():
            if "error" in res:
                print(f"  {sym}: ERROR - {res['error']}")
                continue
            mc = res.get("monte_carlo", {})
            print(f"\n  {sym}:")
            print(f"    Windows: {res['n_windows']}  |  Trades: {res['total_trades']}")
            print(f"    Sharpe: {res['mean_sharpe']:.2f}  |  Return: {res['mean_return_pct']:.1f}%  |  Win Rate: {res['mean_win_rate']:.1f}%")
            print(f"    Monte Carlo: Max DD {mc.get('worst_dd_pct', 0):.1f}%  |  Ruin {mc.get('ruin_probability', 0):.1f}%  |  P10 Eq ${mc.get('p10_equity', 0):.0f}")
        print("=" * 70)
