import sys
import os
_quant_v2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_quant_v2)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np
from typing import Dict, List, Tuple
from collections import deque

from quant_v2.config.pairs import get_pair_config, list_enabled_symbols
from quant_v2.config.crypto_config import CryptoConfig, PairConfig, ScalpingConfig
from risk.kelly import kelly_fraction


class PortfolioRiskManager:
    def __init__(self, config: CryptoConfig):
        self.config = config
        self.scalping: ScalpingConfig = config.scalping
        self._pair_performance: Dict[str, Dict] = {}
        self._total_exposure: Dict[str, float] = {}
        self._equity: float = config.scalping.initial_capital
        self._trade_history: deque = deque(maxlen=500)

    def update_equity(self, equity: float):
        self._equity = equity

    def allocate_capital(self, equity: float, active_symbols: List[str]) -> Dict[str, float]:
        if not active_symbols:
            return {}

        n = len(active_symbols)
        max_positions = self.scalping.max_concurrent_positions
        available_slots = max(0, max_positions - len(active_symbols))
        if available_slots <= 0:
            return {}

        weights = {}
        for sym in active_symbols:
            perf = self._pair_performance.get(sym, {})
            win_rate = perf.get("win_rate", 0.5)
            avg_win = perf.get("avg_win", 0.5)
            avg_loss = perf.get("avg_loss", 0.4)

            kf = kelly_fraction(win_rate, avg_win, abs(avg_loss)) * 0.5
            kf = max(0.05, min(kf, 0.15))
            weights[sym] = kf

        total_weight = sum(weights.values()) or 1.0
        allocation = {}
        for sym, w in weights.items():
            allocation[sym] = w / total_weight

        return allocation

    def calculate_position_size(self, symbol: str, equity: float,
                                entry_price: float, atr: float,
                                risk_pct: float = None,
                                ml_conf: float = 1.0,
                                mtf_conf: float = 1.0,
                                sentiment: float = 0.0,
                                zscore_abs: float = 0.0,
                                rl_conf: float = 1.0) -> float:
        if risk_pct is None:
            risk_pct = self.scalping.account_risk_pct

        pair_config = get_pair_config(symbol)

        stat_conf = min(zscore_abs / 3.0, 1.0) if zscore_abs > 0 else 0.5

        w = self.config.ml
        ensemble = (w.ensemble_weight_stat * stat_conf +
                    w.ensemble_weight_ml * ml_conf +
                    w.ensemble_weight_sentiment * max(0, (1.0 + sentiment) / 2.0) if sentiment != 0 else 0.5 * w.ensemble_weight_sentiment +
                    w.ensemble_weight_mtf * mtf_conf +
                    0.15 * rl_conf)
        ensemble = max(0.2, min(1.5, ensemble))

        effective_risk = risk_pct * ensemble * self._kelly_scaling(symbol)

        total_exposed = sum(self._total_exposure.values()) / max(equity, 1.0)
        max_total_exposure = self.scalping.max_total_exposure_pct if hasattr(self.scalping, 'max_total_exposure_pct') else 0.60
        if total_exposed + effective_risk > max_total_exposure:
            available = max(0.01, max_total_exposure - total_exposed)
            effective_risk = min(effective_risk, available)

        risk_amount = equity * effective_risk

        if atr < 1e-10:
            atr = entry_price * 0.002

        position_size_in_quote = risk_amount / (atr * pair_config.atr_multiplier_sl)

        if entry_price > 0:
            position_size_in_base = position_size_in_quote / entry_price
        else:
            position_size_in_base = position_size_in_quote

        max_notional_pct = 0.20
        min_qty = pair_config.min_qty
        qty_step = pair_config.qty_step
        position_size_in_base = max(min_qty, min(position_size_in_base, equity * max_notional_pct / max(entry_price, 0.01)))
        position_size_in_base = round(position_size_in_base / qty_step) * qty_step
        position_size_in_base = round(position_size_in_base, 8)

        min_notional = pair_config.min_notional
        notional = position_size_in_base * entry_price
        if notional < min_notional:
            position_size_in_base = round(min_notional / max(entry_price, 0.01) / qty_step) * qty_step
            position_size_in_base = max(min_qty, position_size_in_base)

        self._total_exposure[symbol] = position_size_in_base * entry_price

        return position_size_in_base

    def _kelly_scaling(self, symbol: str) -> float:
        perf = self._pair_performance.get(symbol, {})
        recent = [t for t in self._trade_history if t.get("symbol") == symbol]
        if len(recent) < 10:
            return 1.0

        trades = list(recent)[-50:]
        wins = [t["pnl"] for t in trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) if trades else 0.5
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0

        if avg_loss < 1e-10 or avg_win < 1e-10:
            return 1.0

        kf = kelly_fraction(win_rate, avg_win, avg_loss)
        kf = max(0.3, min(kf, 1.5))

        return kf

    def release_position(self, symbol: str):
        self._total_exposure.pop(symbol, None)

    def get_max_positions_per_pair(self) -> int:
        return self.scalping.max_positions_per_pair

    def get_max_concurrent_positions(self) -> int:
        return self.scalping.max_concurrent_positions

    def update_pair_performance(self, symbol: str, trades: List[Dict]):
        if not trades:
            return

        pair_trades = [t for t in trades if t.get("symbol") == symbol]
        if not pair_trades:
            pair_trades = trades

        wins = [t["pnl"] for t in pair_trades if t["pnl"] > 0]
        losses = [t["pnl"] for t in pair_trades if t["pnl"] <= 0]
        n = len(pair_trades)

        self._pair_performance[symbol] = {
            "win_rate": len(wins) / n if n > 0 else 0.5,
            "avg_win": np.mean(wins) if wins else 0.5,
            "avg_loss": abs(np.mean(losses)) if losses else 0.4,
            "profit_factor": abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 0.0,
            "total_trades": n,
            "total_pnl": sum(t["pnl"] for t in pair_trades),
        }

    def should_trade_pair(self, symbol: str) -> Tuple[bool, str]:
        pair_config = get_pair_config(symbol)
        if not pair_config.enabled:
            return False, "pair_disabled"

        perf = self._pair_performance.get(symbol, {})
        total = perf.get("total_trades", 0)
        if total >= 10:
            wr = perf.get("win_rate", 0)
            if wr < 0.35:
                return False, f"win_rate_too_low_{wr:.0%}"

        return True, "allowed"