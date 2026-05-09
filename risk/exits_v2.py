import numpy as np
import pandas as pd
from typing import Tuple, Optional


def atr_trailing_stop(current_price: float, highest_since_entry: float,
                       atr: float, multiplier: float = 1.5,
                       direction: int = 1) -> Tuple[bool, str, float]:
    trail = highest_since_entry - multiplier * atr if direction == 1 else highest_since_entry + multiplier * atr
    if direction == 1:
        if current_price <= trail:
            return True, "atr_trailing_stop", trail
    else:
        if current_price >= trail:
            return True, "atr_trailing_stop", trail
    return False, "", trail


def zscore_velocity_exit(current_zscore: float, prev_zscore: float,
                          direction: int, threshold: float = 0.3) -> Tuple[bool, str]:
    if prev_zscore == 0:
        return False, ""
    velocity = current_zscore - prev_zscore
    if direction == 1 and velocity < -threshold:
        return True, "zscore_velocity_against"
    if direction == -1 and velocity > threshold:
        return True, "zscore_velocity_against"
    return False, ""


def hurst_regime_exit(current_hurst: float, entry_hurst: float,
                       threshold: float = 0.55) -> Tuple[bool, str]:
    if current_hurst > threshold and entry_hurst < threshold:
        return True, f"hurst_regime_change_{current_hurst:.2f}"
    return False, ""


def partial_profit_exit(current_zscore: float, entry_zscore: float,
                        direction: int, target_z: float = 0.5,
                        entry_taken: bool = False) -> Tuple[bool, str, float]:
    if entry_taken:
        return False, "", 1.0
    progress = abs(entry_zscore) - abs(current_zscore)
    total = abs(entry_zscore) - target_z
    if total <= 0:
        return False, "", 1.0
    pct_progress = progress / total if total > 0 else 0
    if abs(current_zscore) <= target_z and abs(entry_zscore) > target_z:
        return True, "partial_profit_z_0.5", 0.5
    return False, "", 1.0


def zscore_exit(
    current_zscore: float,
    entry_zscore: float,
    target_zscore: float = 0.0,
    secondary_target: float = 0.5,
) -> Tuple[bool, str]:

    if abs(current_zscore) <= abs(target_zscore):
        return True, "zscore_mean"

    if abs(current_zscore) <= abs(secondary_target) and abs(entry_zscore) > abs(secondary_target):
        return True, "zscore_partial"

    return False, ""


def time_stop(
    entry_bar: int,
    current_bar: int,
    max_bars: int = 10,
) -> bool:
    return (current_bar - entry_bar) >= max_bars


def combined_exit(
    current_zscore: float,
    entry_zscore: float,
    bar_index: int,
    entry_bar: int,
    signal_direction: int,
    max_bars: int = 10,
    zscore_stop_long: float = -3.5,
    zscore_stop_short: float = 3.5,
    zscore_target: float = 0.0,
    current_price: float = 0.0,
    highest_since_entry: float = 0.0,
    atr: float = 0.0,
    atr_trailing_mult: float = 1.5,
    prev_zscore: float = 0.0,
    zscore_velocity_threshold: float = 0.3,
    current_hurst: float = 0.5,
    entry_hurst: float = 0.5,
    hurst_exit_threshold: float = 0.55,
) -> Tuple[bool, str, float]:

    if signal_direction == 1:
        if current_zscore <= zscore_stop_long:
            return True, "zscore_stop", zscore_stop_long
    elif signal_direction == -1:
        if current_zscore >= zscore_stop_short:
            return True, "zscore_stop", zscore_stop_short

    if atr > 0 and current_price > 0 and highest_since_entry > 0:
        stopped, reason, trail = atr_trailing_stop(
            current_price, highest_since_entry, atr, atr_trailing_mult, signal_direction
        )
        if stopped:
            return True, reason, trail

    if current_hurst > 0 and entry_hurst > 0:
        hurst_exit, hurst_reason = hurst_regime_exit(current_hurst, entry_hurst, hurst_exit_threshold)
        if hurst_exit:
            return True, hurst_reason, current_zscore

    if prev_zscore != 0:
        vel_exit, vel_reason = zscore_velocity_exit(
            current_zscore, prev_zscore, signal_direction, zscore_velocity_threshold
        )
        if vel_exit:
            return True, vel_reason, current_zscore

    should_partial, partial_reason, size_pct = partial_profit_exit(
        current_zscore, entry_zscore, signal_direction, target_z=0.5
    )
    if should_partial and size_pct <= 0.5:
        return True, partial_reason, current_zscore

    should_exit, reason = zscore_exit(current_zscore, entry_zscore, target_zscore=zscore_target, secondary_target=0.5)
    if should_exit:
        return True, reason, current_zscore

    if time_stop(entry_bar, bar_index, max_bars):
        return True, "time_stop", current_zscore

    return False, "", current_zscore


def apply_exits_to_df(
    signals_df: pd.DataFrame,
    max_bars: int = 10,
    zscore_stop_long: float = -3.5,
    zscore_stop_short: float = 3.5,
    atr_trailing_mult: float = 1.5,
    hurst_exit_threshold: float = 0.55,
) -> pd.DataFrame:

    df = signals_df.copy()
    df["exit_signal"] = 0
    df["exit_reason"] = ""
    df["trade_active"] = False

    entry_bar = -1
    entry_zscore = 0.0
    direction = 0
    highest = 0.0
    entry_hurst = 0.5

    col_close = "close" if "close" in df.columns else None
    col_atr = "atr" if "atr" in df.columns else None
    col_hurst = "hurst" if "hurst" in df.columns else None

    for i in range(len(df)):
        if df["signal"].iloc[i] != 0 and not (df["trade_active"].iloc[i - 1] if i > 0 else True):
            entry_bar = i
            entry_zscore = df["zscore"].iloc[i]
            direction = df["signal"].iloc[i]
            if col_close:
                highest = df[col_close].iloc[i]
            entry_hurst = df["hurst"].iloc[i] if col_hurst else 0.5
            df.loc[df.index[i], "trade_active"] = True
            continue

        if not (df["trade_active"].iloc[i - 1] if i > 0 else True):
            continue

        df.loc[df.index[i], "trade_active"] = True
        current_z = df["zscore"].iloc[i]
        prev_z = df["zscore"].iloc[i - 1] if i > 0 else 0.0
        current_hurst = df["hurst"].iloc[i] if col_hurst else 0.5

        close_price = df[col_close].iloc[i] if col_close else 0
        if close_price > 0:
            highest = max(highest, close_price) if direction == 1 else min(highest, close_price)

        atr_val = df[col_atr].iloc[i] if col_atr else 0

        should_exit, reason, _ = combined_exit(
            current_zscore=current_z,
            entry_zscore=entry_zscore,
            bar_index=i,
            entry_bar=entry_bar,
            signal_direction=direction,
            max_bars=max_bars,
            zscore_stop_long=zscore_stop_long,
            zscore_stop_short=zscore_stop_short,
            current_price=close_price,
            highest_since_entry=highest,
            atr=atr_val,
            atr_trailing_mult=atr_trailing_mult,
            prev_zscore=prev_z,
            current_hurst=current_hurst,
            entry_hurst=entry_hurst,
            hurst_exit_threshold=hurst_exit_threshold,
        )

        if should_exit:
            df.loc[df.index[i], "exit_signal"] = -direction
            df.loc[df.index[i], "exit_reason"] = reason
            df.loc[df.index[i], "trade_active"] = False
            highest = 0.0

    return df