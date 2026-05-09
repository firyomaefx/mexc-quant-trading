import sys

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from collections import deque
import time

from training.trading_env import TradingEnv, HAS_GYM


class RLAgent:
    def __init__(self, state_dim: int = 10, lookback: int = 50,
                 model_path: str = None):
        self.state_dim = state_dim
        self.lookback = lookback
        self.model_path = model_path or "quant_v2/training/rl_model.zip"
        self.model = None
        self.env = None
        self._fitted = False
        self._trade_history: deque = deque(maxlen=200)
        self._last_action_probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    def train(self, price_data: np.ndarray, total_timesteps: int = 50000,
              verbose: int = 0) -> Dict:
        if not HAS_GYM:
            print("RL: gymnasium not installed. Install with: pip install gymnasium stable-baselines3")
            return {"error": "gymnasium_not_installed"}

        try:
            from stable_baselines3 import PPO
        except ImportError:
            print("RL: stable-baselines3 not installed. Install with: pip install stable-baselines3")
            return {"error": "stable_baselines3_not_installed"}

        env_data = self._prepare_env_data(price_data)
        self.env = TradingEnv(data=env_data, initial_capital=160.0, lookback=self.lookback)

        try:
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: self.env])

            self.model = PPO(
                "MlpPolicy", vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=verbose,
            )

            self.model.learn(total_timesteps=total_timesteps)
            self._fitted = True

            self.model.save(self.model_path.replace(".zip", ""))
            print(f"RL model trained and saved to {self.model_path}")

            return {
                "status": "trained",
                "timesteps": total_timesteps,
                "model_path": self.model_path,
            }
        except Exception as e:
            print(f"RL training error: {e}")
            return {"error": str(e)}

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[int, np.ndarray]:
        if not self._fitted or self.model is None:
            action = 0
            probs = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
            return action, probs

        try:
            action, _states = self.model.predict(observation, deterministic=deterministic)
            if isinstance(action, np.ndarray):
                action = int(action.item())
            action = max(0, min(4, action))
            probs = np.array([0.4, 0.15, 0.15, 0.15, 0.15])

            try:
                import torch
                with torch.no_grad():
                    probs = self.model.policy.get_distribution(observation.reshape(1, -1) if observation.ndim == 1 else observation.reshape(1, *observation.shape)).distribution.probs.numpy()
            except Exception:
                pass

            if isinstance(probs, np.ndarray) and probs.shape[-1] == 5:
                self._last_action_probs = probs.flatten()
            else:
                self._last_action_probs = self._default_probs(action)

            return action, self._last_action_probs
        except Exception:
            return 0, np.array([0.4, 0.15, 0.15, 0.15, 0.15])

    def should_trade(self, observation: np.ndarray, base_signal: int) -> Tuple[bool, float, str]:
        if not self._fitted:
            return True, 1.0, "rl_not_trained"

        action, probs = self.predict(observation)

        trade_actions = {1: "long", 2: "short"}
        hold_actions = {0: "hold", 3: "close_long", 4: "close_short"}

        if base_signal == 1 and action == 1:
            confidence = float(probs[1])
            return True, confidence, "rl_confirms_long"
        elif base_signal == -1 and action == 2:
            confidence = float(probs[2])
            return True, confidence, "rl_confirms_short"
        elif base_signal == 1 and action == 4:
            return False, float(probs[4]), "rl_suggests_close_long"
        elif base_signal == -1 and action == 3:
            return False, float(probs[3]), "rl_suggests_close_short"
        elif action == 0 and base_signal != 0:
            confidence = float(probs[0])
            return confidence < 0.6, 1.0 - confidence, "rl_suggests_hold"

        return True, 1.0, "rl_neutral"

    def get_action_confidence(self, observation: np.ndarray, action: int) -> float:
        if not self._fitted:
            return 0.5
        _, probs = self.predict(observation)
        if 0 <= action < len(probs):
            return float(probs[action])
        return 0.5

    def update_trade_history(self, pnl: float, action: int, observation: np.ndarray = None):
        self._trade_history.append({
            "pnl": pnl,
            "action": action,
            "timestamp": time.time(),
        })

    def maybe_retrain(self, price_data: np.ndarray, min_trades: int = 200) -> bool:
        if len(self._trade_history) < min_trades:
            return False

        recent_pnls = [t["pnl"] for t in list(self._trade_history)[-100:]]
        win_rate = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls) if recent_pnls else 0

        if win_rate < 0.40:
            print(f"RL retraining triggered (win_rate={win_rate:.1%})...")
            return self.train(price_data, total_timesteps=20000, verbose=0).get("status") == "trained"

        return False

    def load(self, path: str = None) -> bool:
        if not HAS_GYM:
            return False

        path = path or self.model_path
        try:
            from stable_baselines3 import PPO
            path_no_ext = path.replace(".zip", "")
            self.model = PPO.load(path_no_ext)
            self._fitted = True
            print(f"RL model loaded from {path}")
            return True
        except Exception as e:
            print(f"RL model load failed: {e}")
            return False

    def _prepare_env_data(self, price_data: np.ndarray) -> np.ndarray:
        if price_data.ndim == 1:
            data = np.zeros((len(price_data), 5))
            data[:, 3] = price_data
            for i in range(len(price_data)):
                if i > 0:
                    data[i, 2] = max(data[i, 3], data[i - 1, 3])
                    data[i, 1] = min(data[i, 3], data[i - 1, 3])
                    data[i, 0] = data[i - 1, 3]
                else:
                    data[i, 0] = price_data[i]
                    data[i, 1] = price_data[i]
                    data[i, 2] = price_data[i]
                data[i, 4] = np.random.uniform(100, 1000)
            return data
        return price_data

    @staticmethod
    def build_observation(features_df: pd.DataFrame, lookback: int = 50) -> np.ndarray:
        if features_df is None or len(features_df) < 2:
            return np.zeros((lookback, 10), dtype=np.float32)

        close = features_df["close"].values if "close" in features_df.columns else features_df.iloc[:, 0].values
        high = features_df["high"].values if "high" in features_df.columns else close
        low = features_df["low"].values if "low" in features_df.columns else close
        volume = features_df["volume"].values if "volume" in features_df.columns else np.ones(len(close))
        zscore = features_df["zscore"].values if "zscore" in features_df.columns else np.zeros(len(close))
        hurst = features_df["hurst"].values if "hurst" in features_df.columns else np.full(len(close), 0.5)

        n = len(close)
        start = max(0, n - lookback)
        obs = np.zeros((lookback, 10), dtype=np.float32)

        window_close = close[start:n]
        window_high = high[start:n]
        window_low = low[start:n]
        window_vol = volume[start:n]
        window_z = zscore[start:n]
        window_h = hurst[start:n]
        wlen = n - start

        returns = np.diff(window_close, prepend=window_close[0]) / np.where(np.abs(window_close) > 1e-10, window_close, 1.0)
        sma = np.convolve(window_close, np.ones(5) / 5, mode="same")

        obs[:wlen, 0] = returns.astype(np.float32)
        obs[:wlen, 1] = np.where(np.abs(window_close) > 0, (window_close - sma) / np.where(np.abs(sma) > 1e-10, sma, 1.0), 0).astype(np.float32)
        obs[:wlen, 2] = np.where(np.abs(window_close) > 0, (window_high - window_low) / np.where(np.abs(window_close) > 1e-10, window_close, 1.0), 0).astype(np.float32)
        vol_sma = np.convolve(window_vol, np.ones(10) / 10, mode="same")
        obs[:wlen, 3] = np.where(vol_sma > 0, window_vol / vol_sma, 1.0).astype(np.float32)
        obs[:wlen, 4] = (window_close / max(window_close[-1], 1e-10)).astype(np.float32)
        obs[:wlen, 7] = window_z.astype(np.float32)[-wlen:] if wlen <= len(window_z) else np.zeros(wlen, dtype=np.float32)
        obs[:wlen, 8] = window_h.astype(np.float32)[-wlen:] if wlen <= len(window_h) else np.full(wlen, 0.5, dtype=np.float32)
        obs[:wlen, 9] = window_close.astype(np.float32)

        return obs

    def _default_probs(self, action: int) -> np.ndarray:
        probs = np.array([0.4, 0.15, 0.15, 0.15, 0.15])
        probs[action] += 0.3
        probs = probs / probs.sum()
        return probs
