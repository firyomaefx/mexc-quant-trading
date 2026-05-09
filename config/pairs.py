import sys

from config.crypto_config import PairConfig

DEFAULT_PAIRS = {
    "XRP/USDT": PairConfig(
        symbol="XRP/USDT",
        base="XRP",
        quote="USDT",
        min_qty=5.0,
        qty_step=1.0,
        min_notional=5.0,
        pip_value=0.0001,
        typical_spread_pct=0.05,
        zscore_entry_long=-1.8,
        zscore_entry_short=2.0,
        zscore_stop_long=-2.8,
        zscore_stop_short=2.8,
        time_stop_bars=5,
        atr_multiplier_sl=1.5,
        enabled=True,
    ),
    "ADA/USDT": PairConfig(
        symbol="ADA/USDT",
        base="ADA",
        quote="USDT",
        min_qty=10.0,
        qty_step=1.0,
        min_notional=5.0,
        pip_value=0.001,
        typical_spread_pct=0.06,
        zscore_entry_long=-1.8,
        zscore_entry_short=2.0,
        zscore_stop_long=-2.8,
        zscore_stop_short=2.8,
        time_stop_bars=5,
        atr_multiplier_sl=1.5,
        enabled=True,
    ),
    "SOL/USDT": PairConfig(
        symbol="SOL/USDT",
        base="SOL",
        quote="USDT",
        min_qty=0.1,
        qty_step=0.01,
        min_notional=5.0,
        pip_value=0.01,
        typical_spread_pct=0.04,
        zscore_entry_long=-1.8,
        zscore_entry_short=2.0,
        zscore_stop_long=-2.8,
        zscore_stop_short=2.8,
        time_stop_bars=5,
        atr_multiplier_sl=1.5,
        enabled=True,
    ),
}

HIGH_POTENTIAL_PAIRS = {
    "DOGE/USDT": PairConfig(
        symbol="DOGE/USDT",
        base="DOGE",
        quote="USDT",
        min_qty=50.0,
        qty_step=1.0,
        min_notional=5.0,
        pip_value=0.00001,
        typical_spread_pct=0.06,
        zscore_entry_long=-1.6,
        zscore_entry_short=1.8,
        time_stop_bars=4,
        enabled=False,
    ),
    "LTC/USDT": PairConfig(
        symbol="LTC/USDT",
        base="LTC",
        quote="USDT",
        min_qty=0.1,
        qty_step=0.01,
        min_notional=5.0,
        pip_value=0.01,
        typical_spread_pct=0.05,
        zscore_entry_long=-1.8,
        zscore_entry_short=2.0,
        time_stop_bars=5,
        enabled=False,
    ),
    "AVAX/USDT": PairConfig(
        symbol="AVAX/USDT",
        base="AVAX",
        quote="USDT",
        min_qty=0.5,
        qty_step=0.01,
        min_notional=5.0,
        pip_value=0.001,
        typical_spread_pct=0.06,
        zscore_entry_long=-1.8,
        zscore_entry_short=2.0,
        time_stop_bars=5,
        enabled=False,
    ),
}


def get_pair_config(symbol: str) -> PairConfig:
    all_pairs = {**DEFAULT_PAIRS, **HIGH_POTENTIAL_PAIRS}
    if symbol in all_pairs:
        return all_pairs[symbol]
    raise KeyError(f"Pair {symbol} not configured. Available: {list(all_pairs.keys())}")


def list_enabled_symbols(cfg=None) -> list:
    if cfg and cfg.pairs:
        return [s for s, pc in cfg.pairs.items() if pc.enabled]
    return [s for s, pc in DEFAULT_PAIRS.items() if pc.enabled]


def list_all_symbols() -> list:
    all_pairs = {**DEFAULT_PAIRS, **HIGH_POTENTIAL_PAIRS}
    return list(all_pairs.keys())
