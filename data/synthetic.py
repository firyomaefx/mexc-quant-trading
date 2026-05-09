import numpy as np
import pandas as pd
from typing import Optional

class SyntheticDataGenerator:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def generate_ohlcv(
        self,
        n_bars: int = 5000,
        timeframe: int = 5,
        start_price: float = 2000.0,
        volatility: float = 0.001,
        trend: float = 0.0,
        fat_tail_prob: float = 0.03,
        fat_tail_magnitude: float = 2.5,
    ) -> pd.DataFrame:

        returns_base = self.rng.normal(loc=trend, scale=volatility, size=n_bars)
        is_fat_tail = self.rng.random(n_bars) < fat_tail_prob
        tail_shocks = self.rng.standard_t(df=3, size=n_bars) * volatility * fat_tail_magnitude
        returns = np.where(is_fat_tail, returns_base + tail_shocks, returns_base)

        regime_transitions = self.rng.integers(0, 100, n_bars)
        high_vol_mask = regime_transitions < 15
        volatility_regime = np.where(high_vol_mask, volatility * 2.5, volatility)

        v_t = volatility_regime ** 2
        lambda_param = 0.15
        for t in range(1, n_bars):
            v_t[t] = (1 - lambda_param) * v_t[t] + lambda_param * v_t[t - 1]

        sigma = np.sqrt(v_t)
        returns = sigma * (returns / volatility)

        close = start_price * np.exp(np.cumsum(returns))
        open_price = np.roll(close, 1)
        open_price[0] = start_price

        bar_range = 0.65 * sigma * close
        high = np.maximum(open_price, close) + self.rng.uniform(0, bar_range)
        low = np.minimum(open_price, close) - self.rng.uniform(0, bar_range)
        high = np.maximum(high, np.maximum(open_price, close))
        low = np.minimum(low, np.minimum(open_price, close))

        volume = self.rng.lognormal(mean=8.0, sigma=0.8, size=n_bars)

        dt_index = pd.date_range(
            start="2024-01-01 00:00",
            periods=n_bars,
            freq=f"{timeframe}min",
            tz="UTC",
        )

        df = pd.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dt_index)

        return df

    def generate_regime_data(
        self,
        n_bars: int = 5000,
        n_ranging: int = 3000,
        n_trending: int = 2000,
        timeframe: int = 5,
        start_price: float = 2000.0,
    ) -> pd.DataFrame:

        ranging = self.generate_ohlcv(
            n_bars=n_ranging,
            timeframe=timeframe,
            start_price=start_price,
            volatility=0.0008,
            trend=0.0,
            fat_tail_prob=0.02,
        )

        last_close = ranging["close"].iloc[-1]
        trending = self.generate_ohlcv(
            n_bars=n_trending,
            timeframe=timeframe,
            start_price=last_close,
            volatility=0.0015,
            trend=0.0002 if self.rng.random() > 0.5 else -0.0002,
            fat_tail_prob=0.05,
        )

        combined = pd.concat([ranging, trending], ignore_index=True)
        dt_index = pd.date_range(
            start="2024-01-01 00:00",
            periods=len(combined),
            freq=f"{timeframe}min",
            tz="UTC",
        )
        combined.index = dt_index
        combined["regime"] = 0
        combined.iloc[:n_ranging, combined.columns.get_loc("regime")] = 0
        combined.iloc[n_ranging:, combined.columns.get_loc("regime")] = 1

        return combined
