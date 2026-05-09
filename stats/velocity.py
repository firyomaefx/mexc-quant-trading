import numpy as np
from numba import jit


@jit(nopython=True)
def _rolling_ma(series: np.ndarray, window: int) -> np.ndarray:
    n = len(series)
    ma = np.full(n, np.nan)
    for i in range(window - 1, n):
        ma[i] = np.mean(series[i - window + 1 : i + 1])
    return ma


def ma_velocity(series: np.ndarray, ma_period: int = 20, diff_order: int = 1) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]

    if len(series) < ma_period + diff_order:
        return np.full(len(series), np.nan)

    ma = _rolling_ma(series, ma_period)

    for _ in range(diff_order):
        ma = np.gradient(ma)

    return ma


def velocity_approaching_zero(velocity: np.ndarray, epsilon: float = 0.01, lookback: int = 3) -> np.ndarray:
    velocity = np.asarray(velocity, dtype=np.float64)
    n = len(velocity)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(lookback - 1, n):
        recent = velocity[i - lookback + 1 : i + 1]
        recent = recent[np.isfinite(recent)]
        if len(recent) == 0:
            continue
        result[i] = np.all(np.abs(recent) < epsilon)

    return result
