import sys
import os
_quant_v2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_quant_v2)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import deque


class CorrelationFilter:
    def __init__(self, lookback: int = 50, threshold: float = 0.70):
        self.lookback = lookback
        self.threshold = threshold
        self._price_history: Dict[str, deque] = {}
        self._returns_cache: Dict[str, np.ndarray] = {}
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_update: Dict[str, float] = {}

    def update(self, symbol: str, price: float, timestamp: Optional[float] = None):
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.lookback + 1)
        self._price_history[symbol].append(price)

        if len(self._price_history[symbol]) >= 5:
            prices = np.array(list(self._price_history[symbol]))
            rets = np.diff(prices) / np.where(prices[:-1] > 0, prices[:-1], 1.0)
            self._returns_cache[symbol] = rets

        if timestamp:
            self._last_update[symbol] = timestamp

    def compute_correlation_matrix(self) -> pd.DataFrame:
        symbols = list(self._returns_cache.keys())
        valid_returns = {}
        for sym in symbols:
            rets = self._returns_cache.get(sym)
            if rets is not None and len(rets) >= 5:
                valid_returns[sym] = rets[-self.lookback:]

        if len(valid_returns) < 2:
            self._correlation_matrix = pd.DataFrame()
            return self._correlation_matrix

        min_len = min(len(v) for v in valid_returns.values())
        aligned = {}
        for sym, rets in valid_returns.items():
            aligned[sym] = rets[-min_len:]

        df = pd.DataFrame(aligned)
        self._correlation_matrix = df.corr(method="pearson")
        return self._correlation_matrix

    def get_correlation(self, sym1: str, sym2: str) -> float:
        if self._correlation_matrix is None or self._correlation_matrix.empty:
            return 0.0
        try:
            return float(self._correlation_matrix.loc[sym1, sym2])
        except (KeyError, TypeError):
            return 0.0

    def check_same_direction_block(self, target_symbol: str, target_signal: int,
                                   active_positions: List[Dict]) -> Tuple[bool, str]:
        if target_signal == 0:
            return True, "neutral_signal"

        if not active_positions:
            return True, "no_active_positions"

        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix.empty:
            return True, "insufficient_correlation_data"

        for pos in active_positions:
            pos_symbol = pos.get("symbol", "")
            pos_side = pos.get("side", "buy")
            if pos_symbol == target_symbol:
                continue

            pos_direction = 1 if pos_side in ("buy", "long") else -1

            if pos_direction != target_signal:
                continue

            corr = self.get_correlation(target_symbol, pos_symbol)
            if abs(corr) > self.threshold:
                return False, f"corr_blocked_{target_symbol}_{pos_symbol}_{corr:.2f}"

        return True, "corr_passed"

    def get_pair_capital_multiplier(self, target_symbol: str,
                                    active_positions: List[Dict]) -> float:
        if not active_positions:
            return 1.0

        corr_matrix = self.compute_correlation_matrix()
        if corr_matrix.empty:
            return 0.5

        total_corr = 0.0
        count = 0
        for pos in active_positions:
            pos_symbol = pos.get("symbol", "")
            if pos_symbol == target_symbol:
                continue
            corr = abs(self.get_correlation(target_symbol, pos_symbol))
            total_corr += corr
            count += 1

        if count == 0:
            return 1.0

        avg_corr = total_corr / count
        multiplier = 1.0 - avg_corr * 0.5
        return max(0.2, min(1.0, multiplier))

    @staticmethod
    def calculate_returns(symbol: str, close_prices: np.ndarray) -> np.ndarray:
        if len(close_prices) < 2:
            return np.array([])
        returns = np.diff(close_prices) / np.where(close_prices[:-1] > 0, close_prices[:-1], 1.0)
        return returns
