import numpy as np
import pandas as pd
from numba import jit


@jit(nopython=True)
def _rolling_mean_std(series: np.ndarray, window: int) -> tuple:
    n = len(series)
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)

    for i in range(window - 1, n):
        chunk = series[i - window + 1 : i + 1]
        mean[i] = np.mean(chunk)
        std[i] = np.std(chunk)

    return mean, std


def rolling_zscore(
    series: np.ndarray,
    window: int = 100,
) -> pd.DataFrame:
    series = np.asarray(series, dtype=np.float64).ravel()
    series = series[np.isfinite(series)]

    if len(series) < window:
        raise ValueError(f"Series length ({len(series)}) is less than window ({window})")

    mean, std = _rolling_mean_std(series, window)
    mask = ~np.isnan(mean) & ~np.isnan(std)
    zscore = np.full_like(series, np.nan)
    valid = mask
    if np.any(valid):
        with np.errstate(invalid="ignore", divide="ignore"):
            zscore[valid] = np.where(
                std[valid] > 1e-15,
                (series[valid] - mean[valid]) / std[valid],
                0.0,
            )

    return pd.DataFrame({
        "value": series,
        "mean": mean,
        "std": std,
        "zscore": zscore,
    })


def gold_bollinger_bands(
    zscore_df: pd.DataFrame,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    df = zscore_df.copy()
    df["upper_band"] = df["mean"] + multiplier * df["std"]
    df["lower_band"] = df["mean"] - multiplier * df["std"]
    df["band_width"] = df["upper_band"] - df["lower_band"]
    df["price_outside_upper"] = df["value"] > df["upper_band"]
    df["price_outside_lower"] = df["value"] < df["lower_band"]
    return df
