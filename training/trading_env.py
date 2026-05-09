import sys

import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque


try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False


class TradingEnv:
    if HAS_GYM:
        metadata = {"render_modes": ["human"]}

    def __init__(self, data: np.ndarray = None, initial_capital: float = 160.0,
                 commission_pct: float = 0.1, lookback: int = 50):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.lookback = lookback
        self.data = data
        self._setup_spaces()

        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.current_bar = 0
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.done = False
        self._price_history = deque(maxlen=1000)
        self._pnl_history = deque(maxlen=100)

    def _setup_spaces(self):
        if not HAS_GYM:
            self.observation_space = None
            self.action_space = None
            return

        n_features = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.lookback, n_features), dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)

    def set_data(self, data: np.ndarray):
        self.data = data
        self.current_bar = self.lookback

    def reset(self, seed=None, options=None):
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_bar = 0
        self.current_bar = self.lookback
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        self.done = False
        obs = self._get_obs()
        info = self._get_info()
        if HAS_GYM:
            return obs, info
        return obs, info

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        prev_capital = self.capital + self._unrealized_pnl()
        reward = 0.0

        if action == 1 and self.position == 0:
            self._open_position(direction=1)
        elif action == 2 and self.position == 0:
            self._open_position(direction=-1)
        elif action == 3 and self.position == 1:
            reward = self._close_position()
        elif action == 4 and self.position == -1:
            reward = self._close_position()

        self.current_bar += 1

        if self.current_bar >= len(self.data) - 1:
            if self.position != 0:
                self._close_position()
            self.done = True

        if self.position != 0 and self.current_bar - self.entry_bar >= 15:
            reward = self._close_position()

        if self.position != 0:
            unrealized = self._unrealized_pnl()
            if unrealized < -self.initial_capital * 0.03:
                reward = self._close_position()

        current_capital = self.capital + self._unrealized_pnl()
        reward += (current_capital - prev_capital) / self.initial_capital * 10

        if self.position != 0:
            reward -= 0.001

        if abs(reward) > 2.0:
            reward = np.sign(reward) * 2.0

        obs = self._get_obs()
        info = self._get_info()
        terminated = self.done
        truncated = False
        if HAS_GYM:
            return obs, float(reward), terminated, truncated, info
        return obs, float(reward), terminated, info

    def _get_obs(self) -> np.ndarray:
        if self.data is None or self.current_bar < self.lookback:
            return np.zeros((self.lookback, 10), dtype=np.float32)

        start = max(0, self.current_bar - self.lookback)
        end = self.current_bar
        window = self.data[start:end]

        if len(window) < self.lookback:
            pad = np.zeros((self.lookback - len(window), self.data.shape[1]))
            window = np.vstack([pad, window])

        features = np.zeros((self.lookback, 10), dtype=np.float32)
        close = window[:, 3] if window.shape[1] > 3 else window[:, 0]
        high = window[:, 2] if window.shape[1] > 2 else close
        low = window[:, 1] if window.shape[1] > 1 else close
        volume = window[:, 4] if window.shape[1] > 4 else np.ones(len(close))

        returns = np.diff(close, prepend=close[0]) / np.where(np.abs(close) > 1e-10, close, 1.0)
        sma = np.convolve(close, np.ones(5) / 5, mode="same")
        features[:, 0] = returns.astype(np.float32)
        features[:, 1] = np.where(np.abs(close) > 0, (close - sma) / np.where(np.abs(sma) > 1e-10, sma, 1.0), 0).astype(np.float32)
        features[:, 2] = np.where(np.abs(close) > 0, (high - low) / np.where(np.abs(close) > 1e-10, close, 1.0), 0).astype(np.float32)
        vol_sma = np.convolve(volume, np.ones(10) / 10, mode="same")
        features[:, 3] = np.where(vol_sma > 0, volume / vol_sma, 1.0).astype(np.float32)
        features[:, 4] = close / max(close[-1], 1e-10) if close[-1] != 0 else np.ones(self.lookback)
        features[:, 4] = features[:, 4].astype(np.float32)

        pos_onehot = np.zeros(self.lookback, dtype=np.float32)
        if self.position == 1:
            pos_onehot[:] = 1.0
        elif self.position == -1:
            pos_onehot[:] = -1.0
        features[:, 5] = pos_onehot

        bars_held = np.zeros(self.lookback, dtype=np.float32)
        if self.position != 0:
            bars_held[:] = min((self.current_bar - self.entry_bar) / 15.0, 1.0)
        features[:, 6] = bars_held

        unrealized = self._unrealized_pnl()
        pnl_pct = unrealized / self.initial_capital if self.initial_capital > 0 else 0
        pnl_arr = np.zeros(self.lookback, dtype=np.float32)
        pnl_arr[-5:] = pnl_pct
        features[:, 7] = pnl_arr

        features[:, 8] = (self.capital / self.initial_capital - 1.0) if self.initial_capital > 0 else 0

        hour = np.zeros(self.lookback, dtype=np.float32)
        features[:, 9] = hour

        return features

    def _get_info(self) -> Dict:
        return {
            "capital": self.capital,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_count / max(1, self.trade_count),
            "current_bar": self.current_bar,
        }

    def _open_position(self, direction: int):
        if self.data is None or self.current_bar >= len(self.data):
            return
        self.position = direction
        self.entry_price = self.data[self.current_bar, 3] if self.data.shape[1] > 3 else self.data[self.current_bar, 0]
        self.entry_bar = self.current_bar
        commission = self.capital * self.commission_pct / 100.0
        self.capital -= commission

    def _close_position(self) -> float:
        if self.position == 0 or self.data is None:
            return 0.0
        if self.current_bar >= len(self.data):
            self.current_bar = len(self.data) - 1

        exit_price = self.data[self.current_bar, 3] if self.data.shape[1] > 3 else self.data[self.current_bar, 0]
        pnl = (exit_price - self.entry_price) * self.position
        pnl_pct = pnl / self.entry_price if abs(self.entry_price) > 1e-10 else 0
        dollar_pnl = self.capital * pnl_pct
        commission = self.capital * self.commission_pct / 100.0
        self.capital += dollar_pnl - commission
        self.total_pnl += dollar_pnl
        self.trade_count += 1
        if dollar_pnl > 0:
            self.win_count += 1
        self._pnl_history.append(dollar_pnl)
        self.position = 0
        self.entry_price = 0.0
        return dollar_pnl / self.initial_capital

    def _unrealized_pnl(self) -> float:
        if self.position == 0 or self.data is None:
            return 0.0
        if self.current_bar >= len(self.data):
            return 0.0
        current_price = self.data[self.current_bar, 3] if self.data.shape[1] > 3 else self.data[self.current_bar, 0]
        pnl_pct = (current_price - self.entry_price) * self.position / max(abs(self.entry_price), 1e-10)
        return self.capital * pnl_pct
