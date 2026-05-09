import numpy as np
import pandas as pd
from typing import Optional


def volatility_stop(
    entry_price: float,
    entry_zscore: float,
    zscore_threshold_long: float = -3.5,
    zscore_threshold_short: float = 3.5,
    atr: Optional[float] = None,
    atr_multiplier: float = 1.5,
) -> float:

    if atr is None:
        atr = entry_price * 0.005

    zscore_sl_distance = abs(abs(entry_zscore) - abs(zscore_threshold_long)) * atr

    return max(zscore_sl_distance, atr * atr_multiplier)


def initial_stop_distance(
    entry_price: float,
    atr: float,
    multiplier: float = 1.5,
    min_stop_pct: float = 0.002,
) -> float:
    stop_distance = atr * multiplier
    min_stop = entry_price * min_stop_pct
    return max(stop_distance, min_stop)


def trailing_stop(
    zscore_series: np.ndarray,
    improvement_required: float = 1.0,
) -> float:
    if len(zscore_series) < 2:
        return zscore_series[-1] if len(zscore_series) > 0 else 0.0

    current_z = zscore_series[-1]
    best_z = np.max(np.abs(zscore_series))
    trailing = current_z + np.sign(current_z) * improvement_required

    return trailing


def atr_from_df(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ),
    )

    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr
