import numpy as np
import pandas as pd
from typing import Optional, Dict

from stats.hurst import hurst_exponent, rolling_hurst
from stats.zscore import rolling_zscore
from stats.velocity import ma_velocity, velocity_approaching_zero
from stats.hmm import HMMRegimeDetector
from config.settings import ThresholdConfig, WindowConfig


class SignalGenerator:
    def __init__(self, config):
        self.threshold: ThresholdConfig = config.threshold
        self.window: WindowConfig = config.window
        self.hmm: Optional[HMMRegimeDetector] = None
        self._returns: Optional[np.ndarray] = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"].values.astype(np.float64)
        n = len(close)
        features = pd.DataFrame(index=df.index)

        returns = np.diff(close, prepend=close[0]) / np.where(close > 0, close, 1.0)
        self._returns = np.where(np.isfinite(returns), returns, 0.0)

        zscore_df = rolling_zscore(close, window=self.window.rolling_zscore)
        features["zscore"] = zscore_df["zscore"].values
        features["mean"] = zscore_df["mean"].values
        features["std"] = zscore_df["std"].values

        h_vals = rolling_hurst(close, window=self.window.rolling_zscore, max_lag=self.window.hurst_max_lag)
        features["hurst"] = h_vals

        velocity = ma_velocity(close, ma_period=self.window.rolling_ma)
        features["velocity"] = velocity
        features["velocity_zero"] = velocity_approaching_zero(
            velocity, epsilon=self.threshold.velocity_epsilon
        ).astype(int)

        try:
            if self.hmm is None:
                self.hmm = HMMRegimeDetector(n_states=2)
                self.hmm.fit(self._returns)
        except Exception:
            pass

        if self.hmm is not None and self.hmm._fitted:
            features["hmm_ranging_prob"] = self.hmm.predict_proba_series(self._returns)
            features["hmm_state"] = 0
        else:
            features["hmm_ranging_prob"] = 0.9
            features["hmm_state"] = 0

        features["returns"] = self._returns
        return features

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        df = features.copy()

        is_mean_revert = df["hurst"] < self.threshold.hurst_mean_revert
        is_oversold = df["zscore"] < self.threshold.zscore_entry_long
        is_overbought = df["zscore"] > self.threshold.zscore_entry_short
        velocity_flat = df["velocity_zero"] == 1

        long_condition = is_mean_revert & is_oversold & velocity_flat
        short_condition = is_mean_revert & is_overbought & velocity_flat

        if "hmm_ranging_prob" in df.columns and self.threshold.hmm_ranging_prob > 0.01:
            hmm_ranging = df["hmm_ranging_prob"] >= self.threshold.hmm_ranging_prob
            long_condition = long_condition & hmm_ranging
            short_condition = short_condition & hmm_ranging

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

        df["entry_zscore"] = df["zscore"].where(df["signal"] != 0)

        return df

    def compute_and_generate(self, df: pd.DataFrame) -> pd.DataFrame:
        features = self.compute_features(df)
        return self.generate_signals(features)
