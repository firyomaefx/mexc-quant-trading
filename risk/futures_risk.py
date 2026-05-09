import os
import sys
import sys

import time
from typing import Dict, Tuple, Optional, List
from datetime import datetime

from config.crypto_config import FuturesConfig


class FuturesRiskManager:
    def __init__(self, config: FuturesConfig = None):
        self.config = config or FuturesConfig()
        self._liq_warnings: Dict[str, float] = {}
        self._funding_alerts: Dict[str, float] = {}

    def check_entry_risk(self, entry_price: float, liquidation_price: float,
                         side: str, leverage: int) -> Tuple[bool, str, float]:
        if not self.config.enabled:
            return True, "futures_disabled", 1.0

        if leverage > self.config.max_leverage:
            return False, f"leverage_too_high_{leverage}x", 0.0

        if entry_price <= 0 or liquidation_price <= 0:
            return True, "no_liquidation_price", 1.0

        if side in ("buy", "long"):
            if liquidation_price >= entry_price:
                return False, f"invalid_liq_price_long_{liquidation_price}", 0.0
            distance_pct = (entry_price - liquidation_price) / entry_price
        else:
            if liquidation_price <= entry_price:
                return False, f"invalid_liq_price_short_{liquidation_price}", 0.0
            distance_pct = (liquidation_price - entry_price) / entry_price

        safe_threshold = self.config.liq_safety_pct / leverage
        if distance_pct < safe_threshold:
            return False, f"liq_too_close_{distance_pct:.2%}", distance_pct

        if distance_pct < safe_threshold * 2:
            return True, f"liq_warning_{distance_pct:.2%}", 0.5
        else:
            return True, f"liq_safe_{distance_pct:.2%}", 1.0

    def monitor_position(self, symbol: str, current_price: float,
                         liquidation_price: float, side: str,
                         leverage: int) -> Tuple[bool, str]:
        if current_price <= 0 or liquidation_price <= 0:
            return False, "ok"

        if side in ("buy", "long"):
            distance_pct = (current_price - liquidation_price) / current_price
        else:
            distance_pct = (liquidation_price - current_price) / current_price

        danger_threshold = self.config.liq_safety_pct / leverage
        critical_threshold = danger_threshold * 0.5

        now = time.time()
        last_warn = self._liq_warnings.get(symbol, 0)

        if distance_pct <= critical_threshold:
            self._liq_warnings[symbol] = now
            return True, f"LIQUIDATION_CRITICAL_{symbol}_{distance_pct:.2%}"

        if distance_pct <= danger_threshold:
            if now - last_warn > 30:
                self._liq_warnings[symbol] = now
                return True, f"liquidation_warning_{symbol}_{distance_pct:.2%}"
            return False, "ok"

        return False, "ok"

    def check_funding_rate(self, symbol: str, funding_rate: float) -> Tuple[bool, str]:
        if abs(funding_rate) > self.config.funding_rate_limit:
            now = time.time()
            last_alert = self._funding_alerts.get(symbol, 0)
            if now - last_alert > 3600:
                self._funding_alerts[symbol] = now
                direction = "longs_paying" if funding_rate > 0 else "shorts_paying"
                return True, f"high_funding_{symbol}_{funding_rate:.4%}_{direction}"
        return False, "ok"

    def calculate_max_position_size(self, equity: float, entry_price: float,
                                    leverage: int, risk_pct: float = 0.015) -> float:
        if not self.config.enabled:
            return 0.0

        effective_equity = equity * risk_pct
        position_value = effective_equity * leverage
        max_position = min(position_value, equity * leverage * 0.5)

        return max_position / max(entry_price, 0.01)

    def get_leverage_for_volatility(self, atr_pct: float) -> int:
        if atr_pct < 0.001:
            return self.config.max_leverage
        elif atr_pct < 0.003:
            return max(1, self.config.max_leverage - 1)
        elif atr_pct < 0.006:
            return max(1, self.config.max_leverage - 2)
        else:
            return 1

    def get_margin_required(self, position_value: float, leverage: int) -> float:
        return position_value / leverage

    def is_margin_sufficient(self, available: float, required: float,
                             buffer_pct: float = 0.3) -> bool:
        return available >= required * (1.0 + buffer_pct)