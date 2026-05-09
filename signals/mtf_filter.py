import os
import sys
import sys

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


class MTFFilter:
    def __init__(self, adx_trend_threshold: float = 25.0,
                 ema_trend_period: int = 20,
                 rsi_overbought: float = 70.0,
                 rsi_oversold: float = 30.0):
        self.adx_trend_threshold = adx_trend_threshold
        self.ema_trend_period = ema_trend_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold

    def evaluate(self, primary_df: pd.DataFrame,
                 secondary_df: Optional[pd.DataFrame] = None,
                 tertiary_df: Optional[pd.DataFrame] = None,
                 signal: int = 0) -> Tuple[bool, float, str]:

        if signal == 0:
            return True, 1.0, "neutral"

        reasons = []
        confidence = 1.0

        if tertiary_df is not None and len(tertiary_df) >= 50:
            adx_pass, adx_conf, adx_reason = self._adx_filter(tertiary_df, signal)
            confidence *= adx_conf
            if not adx_pass:
                reasons.append(adx_reason)

        if secondary_df is not None and len(secondary_df) >= self.ema_trend_period + 5:
            trend_pass, trend_conf, trend_reason = self._trend_filter(secondary_df, signal)
            confidence *= trend_conf
            if not trend_pass:
                reasons.append(trend_reason)

        if primary_df is not None and len(primary_df) >= 20:
            rsi_pass, rsi_conf, rsi_reason = self._rsi_filter(primary_df, signal)
            confidence *= rsi_conf
            if not rsi_pass:
                reasons.append(rsi_reason)

            vol_pass, vol_conf, vol_reason = self._volume_filter(primary_df)
            confidence *= vol_conf
            if not vol_pass:
                reasons.append(vol_reason)

        passed = confidence >= 0.3
        reason = "mtf_passed" if passed else "; ".join(reasons)
        return passed, confidence, reason

    def _adx_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, float, str]:
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        close = df["close"].values.astype(np.float64)
        adx = self._compute_adx(high, low, close, period=14)

        if np.isnan(adx):
            return True, 1.0, "adx_nan"

        if adx > self.adx_trend_threshold:
            return False, 0.3, f"trending_adx_{adx:.0f}"
        elif adx < 15:
            return True, 1.1, f"ranging_adx_{adx:.0f}"
        else:
            return True, 1.0, f"neutral_adx_{adx:.0f}"

    def _trend_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, float, str]:
        close = df["close"].values.astype(np.float64)
        ema = self._compute_ema(close, self.ema_trend_period)

        if len(ema) < 5:
            return True, 1.0, "ema_insufficient"

        slope = (ema[-1] - ema[-5]) / max(abs(ema[-5]), 1e-10)
        slope_pct = slope * 100.0

        if signal == 1 and slope_pct < -0.05:
            return False, 0.5, "trend_down_long_block"
        elif signal == -1 and slope_pct > 0.05:
            return False, 0.5, "trend_up_short_block"
        elif signal == 1 and slope_pct > 0.15:
            return True, 1.2, "trend_aligned_long"
        elif signal == -1 and slope_pct < -0.15:
            return True, 1.2, "trend_aligned_short"
        else:
            return True, 1.0, "trend_neutral"

    def _rsi_filter(self, df: pd.DataFrame, signal: int) -> Tuple[bool, float, str]:
        close = df["close"].values.astype(np.float64)
        rsi = self._compute_rsi(close, period=14)

        if np.isnan(rsi):
            return True, 1.0, "rsi_nan"

        if signal == 1 and rsi > self.rsi_overbought:
            return False, 0.4, f"rsi_overbought_{rsi:.0f}"
        elif signal == -1 and rsi < self.rsi_oversold:
            return False, 0.4, f"rsi_oversold_{rsi:.0f}"
        elif signal == 1 and rsi < 35:
            return True, 1.1, f"rsi_oversold_long_{rsi:.0f}"
        elif signal == -1 and rsi > 65:
            return True, 1.1, f"rsi_overbought_short_{rsi:.0f}"
        else:
            return True, 1.0, f"rsi_neutral_{rsi:.0f}"

    def _volume_filter(self, df: pd.DataFrame) -> Tuple[bool, float, str]:
        if "volume" not in df.columns:
            return True, 1.0, "no_volume"

        vol = df["volume"].values.astype(np.float64)
        if len(vol) < 10:
            return True, 1.0, "vol_insufficient"

        recent_avg = np.mean(vol[-5:])
        prior_avg = np.mean(vol[-15:-5])

        if prior_avg <= 0:
            return True, 1.0, "vol_zero"

        ratio = recent_avg / prior_avg
        if ratio < 0.3:
            return False, 0.5, f"volume_dried_{ratio:.2f}"
        elif ratio > 3.0:
            return True, 0.8, f"volume_spike_{ratio:.2f}"
        else:
            return True, 1.0, f"volume_normal_{ratio:.2f}"

    @staticmethod
    def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int = 14) -> float:
        n = len(close)
        if n < period + 1:
            return float("nan")

        tr = np.maximum(high[1:] - low[1:],
                        np.abs(high[1:] - close[:-1]),
                        np.abs(low[1:] - close[:-1]))
        atr = np.mean(tr[:period]) if len(tr) >= period else np.mean(tr)

        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                            np.maximum(low[:-1] - low[1:], 0), 0)

        if atr < 1e-10:
            return 0.0

        di_plus = np.mean(dm_plus[-period:]) / atr * 100
        di_minus = np.mean(dm_minus[-period:]) / atr * 100
        dx = abs(di_plus - di_minus) / max(di_plus + di_minus, 1e-10) * 100
        return dx

    @staticmethod
    def _compute_ema(data: np.ndarray, period: int) -> np.ndarray:
        n = len(data)
        if n < period:
            return np.array([data[-1]] if n > 0 else [0])
        alpha = 2.0 / (period + 1.0)
        ema = np.zeros(n)
        ema[period - 1] = np.mean(data[:period])
        for i in range(period, n):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        n = len(close)
        if n < period + 1:
            return 50.0
        deltas = np.diff(close[-period - 1:])
        gains = np.sum(np.where(deltas > 0, deltas, 0))
        losses = np.sum(np.where(deltas < 0, -deltas, 0))
        avg_gain = gains / period
        avg_loss = losses / period
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)