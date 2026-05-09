import numpy as np
import pandas as pd
from typing import Tuple, Optional


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
    max_bars: int = 3,
) -> bool:
    return (current_bar - entry_bar) >= max_bars


def combined_exit(
    current_zscore: float,
    entry_zscore: float,
    bar_index: int,
    entry_bar: int,
    signal_direction: int,
    max_bars: int = 3,
    zscore_stop_long: float = -3.5,
    zscore_stop_short: float = 3.5,
    zscore_target: float = 0.0,
) -> Tuple[bool, str, float]:

    if signal_direction == 1:
        if current_zscore <= zscore_stop_long:
            return True, "zscore_stop", zscore_stop_long
    elif signal_direction == -1:
        if current_zscore >= zscore_stop_short:
            return True, "zscore_stop", zscore_stop_short

    if time_stop(entry_bar, bar_index, max_bars):
        return True, "time_stop", current_zscore

    should_exit, reason = zscore_exit(current_zscore, entry_zscore, target_zscore=zscore_target)
    if should_exit:
        return True, reason, current_zscore

    return False, "", current_zscore


def apply_exits_to_df(
    signals_df: pd.DataFrame,
    max_bars: int = 3,
    zscore_stop_long: float = -3.5,
    zscore_stop_short: float = 3.5,
) -> pd.DataFrame:

    df = signals_df.copy()
    df["exit_signal"] = 0
    df["exit_reason"] = ""
    df["trade_active"] = False

    entry_bar = -1
    entry_zscore = 0.0
    direction = 0

    for i in range(len(df)):
        if df["signal"].iloc[i] != 0 and not df["trade_active"].iloc[i - 1] if i > 0 else True:
            entry_bar = i
            entry_zscore = df["zscore"].iloc[i]
            direction = df["signal"].iloc[i]
            df.loc[df.index[i], "trade_active"] = True
            continue

        if not df["trade_active"].iloc[i - 1] if i > 0 else True:
            continue

        df.loc[df.index[i], "trade_active"] = True
        current_z = df["zscore"].iloc[i]

        should_exit, reason, _ = combined_exit(
            current_zscore=current_z,
            entry_zscore=entry_zscore,
            bar_index=i,
            entry_bar=entry_bar,
            signal_direction=direction,
            max_bars=max_bars,
            zscore_stop_long=zscore_stop_long,
            zscore_stop_short=zscore_stop_short,
        )

        if should_exit:
            df.loc[df.index[i], "exit_signal"] = -direction
            df.loc[df.index[i], "exit_reason"] = reason
            df.loc[df.index[i], "trade_active"] = False

    return df
