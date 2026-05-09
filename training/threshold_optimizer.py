import os
import sys
import sys

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from collections import deque
import time

from config.settings import ThresholdConfig, WindowConfig, GoldConfig
from signals.generator import SignalGenerator
from risk.exits import apply_exits_to_df
from risk.stops import atr_from_df


class ThresholdOptimizer:
    def __init__(self, n_trials: int = 100, timeout_seconds: int = 300):
        self.n_trials = n_trials
        self.timeout_seconds = timeout_seconds
        self.best_params: Dict = {}
        self.best_sharpe: float = -999.0
        self._results: list = []

    def optimize(self, df: pd.DataFrame, symbol: str = "XRP/USDT") -> Dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("ThresholdOptimizer: optuna not installed. Using grid search fallback.")
            return self._grid_search(df, symbol)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: self._objective(trial, df, symbol),
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            show_progress_bar=False,
        )

        self.best_params = study.best_params
        self.best_sharpe = study.best_value

        return {
            "best_params": self.best_params,
            "best_sharpe": self.best_sharpe,
            "n_trials": len(study.trials),
            "symbol": symbol,
        }

    def _objective(self, trial, df: pd.DataFrame, symbol: str) -> float:
        hurst_thresh = trial.suggest_float("hurst_mean_revert", 0.25, 0.50)
        zscore_long = trial.suggest_float("zscore_entry_long", -3.0, -1.2)
        zscore_short = trial.suggest_float("zscore_entry_short", 1.2, 3.5)
        zscore_stop_long = trial.suggest_float("zscore_stop_long", -4.5, -2.5)
        zscore_stop_short = trial.suggest_float("zscore_stop_short", 2.5, 4.5)
        time_stop = trial.suggest_int("time_stop_bars", 5, 20)
        velocity_eps = trial.suggest_float("velocity_epsilon", 1.0, 5.0)
        atr_trail = trial.suggest_float("atr_trailing_mult", 1.0, 3.0)
        prob_thresh = trial.suggest_float("prob_threshold", 0.40, 0.70)

        threshold = ThresholdConfig(
            hurst_mean_revert=hurst_thresh,
            zscore_entry_long=zscore_long,
            zscore_entry_short=zscore_short,
            zscore_stop_long=zscore_stop_long,
            zscore_stop_short=zscore_stop_short,
            velocity_epsilon=velocity_eps,
            time_stop_bars=time_stop,
        )
        window = WindowConfig(rolling_zscore=100, rolling_ma=20, hurst_max_lag=20)

        dummy = GoldConfig()
        dummy.threshold = threshold
        dummy.window = window
        sg = SignalGenerator(dummy)

        try:
            features = sg.compute_and_generate(df)
            if features.empty or "signal" not in features.columns:
                return -10.0

            signals = features["signal"].values
            close = features["close"].values if "close" in features.columns else np.zeros(len(features))

            n_trades = 0
            wins = 0
            total_pnl = 0.0
            pnl_list = []
            position = 0
            entry_price = 0.0
            entry_bar = 0

            for i in range(1, len(signals)):
                if close[i] <= 0:
                    continue
                price = close[i]
                if position == 0 and signals[i] != 0:
                    position = int(signals[i])
                    entry_price = price
                    entry_bar = i
                elif position != 0:
                    bars_held = i - entry_bar
                    z = features["zscore"].values[i] if "zscore" in features.columns else 0
                    exit = False
                    if bars_held >= time_stop:
                        exit = True
                    if position == 1 and z <= zscore_stop_long:
                        exit = True
                    if position == -1 and z >= zscore_stop_short:
                        exit = True
                    if abs(z) < 0.3 and abs(features["zscore"].values[entry_bar] if entry_bar < len(features) else 0) > 1.0:
                        exit = True
                    if exit:
                        pnl = (price - entry_price) * position
                        total_pnl += pnl
                        pnl_list.append(pnl)
                        n_trades += 1
                        if pnl > 0:
                            wins += 1
                        position = 0

            if n_trades < 5:
                return -5.0

            win_rate = wins / n_trades if n_trades > 0 else 0
            avg_pnl = total_pnl / n_trades if n_trades > 0 else 0
            sharpe = avg_pnl / (np.std(pnl_list) + 1e-10) if n_trades > 1 else 0

            return sharpe * (1 + n_trades / 100)

        except Exception:
            return -10.0

    def _grid_search(self, df: pd.DataFrame, symbol: str) -> Dict:
        param_grid = {
            "hurst_mean_revert": [0.30, 0.35, 0.40, 0.45],
            "zscore_entry_long": [-2.5, -2.0, -1.5],
            "zscore_entry_short": [1.5, 2.0, 2.5],
            "time_stop_bars": [5, 10, 15],
        }

        best_sharpe = -999.0
        best_params = {}

        for hurst in param_grid["hurst_mean_revert"]:
            for z_long in param_grid["zscore_entry_long"]:
                for z_short in param_grid["zscore_entry_short"]:
                    for ts in param_grid["time_stop_bars"]:
                        threshold = ThresholdConfig(
                            hurst_mean_revert=hurst,
                            zscore_entry_long=z_long,
                            zscore_entry_short=z_short,
                            time_stop_bars=ts,
                        )
                        window = WindowConfig()
                        dummy = GoldConfig()
                        dummy.threshold = threshold
                        dummy.window = window
                        sg = SignalGenerator(dummy)

                        try:
                            features = sg.compute_and_generate(df)
                            if features.empty:
                                continue
                            sharpe = self._quick_sharpe(features)
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_params = {
                                    "hurst_mean_revert": hurst,
                                    "zscore_entry_long": z_long,
                                    "zscore_entry_short": z_short,
                                    "time_stop_bars": ts,
                                }
                        except Exception:
                            continue

        self.best_params = best_params
        self.best_sharpe = best_sharpe
        return {"best_params": best_params, "best_sharpe": best_sharpe}

    def _quick_sharpe(self, features: pd.DataFrame) -> float:
        if "signal" not in features.columns or "close" not in features.columns:
            return -999.0

        signals = features["signal"].values
        close = features["close"].values

        pnls = []
        position = 0
        entry_price = 0.0

        for i in range(1, len(signals)):
            if position == 0 and signals[i] != 0:
                position = int(signals[i])
                entry_price = close[i]
            elif position != 0:
                z = features["zscore"].values[i] if "zscore" in features.columns else 0
                if abs(z) < 0.3:
                    pnl = (close[i] - entry_price) * position
                    pnls.append(pnl)
                    position = 0

        if len(pnls) < 3:
            return -5.0

        pnls = np.array(pnls)
        mean = np.mean(pnls)
        std = np.std(pnls)
        return mean / std if std > 1e-10 else 0.0

    def save_params(self, path: str = "quant_v2/config/optimized_params.json"):
        import json
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "best_params": self.best_params,
            "best_sharpe": self.best_sharpe,
            "timestamp": datetime.now().isoformat(),
            "n_results": len(self._results),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Optimized params saved to {path}")

    def load_params(self, path: str = "quant_v2/config/optimized_params.json") -> Dict:
        import json
        if not os.path.exists(path):
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}