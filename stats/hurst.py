import numpy as np
from numba import jit


@jit(nopython=True)
def _hurst_var(series: np.ndarray, max_lag: int = 20) -> float:
    n = len(series)
    max_lag = min(max_lag, n // 4)

    if n < 8 or max_lag < 3:
        return 0.5

    lags = np.arange(2, max_lag + 1, dtype=np.int64)
    tau_values = np.empty(len(lags))

    for i, lag in enumerate(lags):
        n_pairs = n - lag
        if n_pairs < 2:
            tau_values[i] = 0.0
            continue

        ss = 0.0
        for j in range(n_pairs):
            diff = series[j + lag] - series[j]
            ss += diff * diff

        tau_values[i] = ss / n_pairs

    valid = tau_values > 1e-20
    n_valid = np.sum(valid)
    if n_valid < 3:
        return 0.5

    log_lags = np.log(lags[valid])
    log_tau = np.log(tau_values[valid])

    x_mean = np.mean(log_lags)
    y_mean = np.mean(log_tau)
    num = np.sum((log_lags - x_mean) * (log_tau - y_mean))
    den = np.sum((log_lags - x_mean) ** 2)
    if den < 1e-15:
        return 0.5

    slope = num / den
    h = slope / 2.0
    return h


def hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    if isinstance(series, (list,)):
        series = np.asarray(series, dtype=np.float64)
    else:
        series = np.asarray(series, dtype=np.float64).ravel()

    series = series[np.isfinite(series)]
    if len(series) < 30:
        return 0.5

    return float(np.clip(_hurst_var(series, max_lag), 0.01, 0.99))


def rolling_hurst(series: np.ndarray, window: int = 100, max_lag: int = 20) -> np.ndarray:
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)
    result = np.full(n, np.nan)

    for i in range(window, n + 1):
        result[i - 1] = hurst_exponent(series[i - window : i], max_lag)

    return result
