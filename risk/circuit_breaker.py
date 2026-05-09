import os
import sys
import sys

import time
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque


class CircuitBreaker:
    def __init__(self, initial_capital: float = 160.0,
                 max_daily_loss_pct: float = 0.05,
                 max_drawdown_pct: float = 0.15,
                 max_consecutive_losses: int = 3,
                 cooldown_seconds: float = 90.0,
                 min_zscore_confidence: float = 1.2,
                 max_spread_pct: float = 0.15,
                 max_volatility_ratio: float = 2.0,
                 trade_min_interval: float = 60.0):
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_seconds = cooldown_seconds
        self.min_zscore_confidence = min_zscore_confidence
        self.max_spread_pct = max_spread_pct
        self.max_volatility_ratio = max_volatility_ratio
        self.trade_min_interval = trade_min_interval

        self._peak_equity = initial_capital
        self._daily_start_equity = initial_capital
        self._daily_pnl = 0.0
        self._daily_date = datetime.now().date()
        self._consecutive_losses = 0
        self._last_trade_time: Dict[str, float] = {}
        self._global_last_trade = 0.0
        self._trades_today = 0
        self._is_halted = False
        self._halt_reason = ""
        self._halt_until = 0.0
        self._trades: deque = deque(maxlen=200)

    def update_equity(self, equity: float):
        self._check_daily_reset()

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = 1.0 - (equity / max(self._peak_equity, 1.0))
        if drawdown >= self.max_drawdown_pct:
            self._halt(f"max_drawdown_{drawdown:.1%}", None)

    def record_trade(self, symbol: str, pnl: float, timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        self._check_daily_reset()

        self._trades.append({"symbol": symbol, "pnl": pnl, "timestamp": ts})
        self._last_trade_time[symbol] = ts
        self._global_last_trade = ts
        self._trades_today += 1
        self._daily_pnl += pnl

        if pnl > 0:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        max_daily_loss = self.initial_capital * self.max_daily_loss_pct
        if self._daily_pnl <= -max_daily_loss:
            self._halt(f"daily_loss_{self._daily_pnl:.2f}", self._end_of_day())

        if self._consecutive_losses >= self.max_consecutive_losses:
            self._halt(f"consecutive_losses_{self._consecutive_losses}", ts + self.cooldown_seconds)

    def check_trade_allowed(self, symbol: str, zscore: float = 0.0,
                            spread_pct: float = 0.0, volatility_ratio: float = 1.0,
                            equity: float = 0.0) -> Tuple[bool, str]:
        if self._is_halted:
            if self._halt_until and time.time() < self._halt_until:
                remaining = self._halt_until - time.time()
                return False, f"halted_{self._halt_reason}_remaining_{remaining:.0f}s"
            elif self._halt_until == 0:
                return False, f"halted_{self._halt_reason}_permanent"
            else:
                self._is_halted = False
                self._halt_reason = ""
                self._halt_until = 0.0
                self._consecutive_losses = 0

        if equity > 0:
            drawdown = 1.0 - (equity / max(self._peak_equity, 1.0))
            if drawdown >= self.max_drawdown_pct:
                return False, f"max_drawdown_{drawdown:.1%}"

        self._check_daily_reset()
        if equity > 0:
            self._daily_start_equity = max(self._daily_start_equity, equity)
        daily_loss_limit = self.initial_capital * self.max_daily_loss_pct
        if self._daily_pnl <= -daily_loss_limit:
            return False, f"daily_loss_limit_{daily_loss_limit:.0f}"

        if abs(zscore) > 0 and abs(zscore) < self.min_zscore_confidence:
            return False, f"zscore_too_low_{abs(zscore):.2f}"

        if spread_pct > self.max_spread_pct:
            return False, f"spread_too_high_{spread_pct:.3f}"

        if volatility_ratio > self.max_volatility_ratio:
            return False, f"volatility_too_high_{volatility_ratio:.2f}"

        now = time.time()
        if symbol in self._last_trade_time:
            elapsed = now - self._last_trade_time[symbol]
            if elapsed < self.trade_min_interval:
                return False, f"trade_interval_{elapsed:.0f}s"
        elif self._global_last_trade > 0:
            elapsed = now - self._global_last_trade
            if elapsed < self.cooldown_seconds:
                return False, f"cooldown_{elapsed:.0f}s"

        if self._consecutive_losses >= self.max_consecutive_losses:
            return False, f"consecutive_losses_{self._consecutive_losses}"

        return True, "allowed"

    def _halt(self, reason: str, until: Optional[float]):
        self._is_halted = True
        self._halt_reason = reason
        self._halt_until = until or 0.0
        print(f"\n[BREAKER] TRADING HALTED: {reason}")
        if until:
            print(f"  Resume at: {datetime.fromtimestamp(until):%H:%M:%S}")

    def _check_daily_reset(self):
        today = datetime.now().date()
        if today != self._daily_date:
            self._daily_date = today
            self._daily_pnl = 0.0
            self._trades_today = 0
            self._consecutive_losses = 0
            self._is_halted = False
            self._halt_reason = ""
            self._halt_until = 0.0
            print(f"\n[NEW DAY] Daily stats reset. Start equity: ${self._daily_start_equity:.2f}")

    def _end_of_day(self) -> float:
        now = datetime.now()
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return tomorrow.timestamp()

    def get_status(self) -> Dict:
        return {
            "is_halted": self._is_halted,
            "halt_reason": self._halt_reason,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": round(self._daily_pnl, 2),
            "daily_pnl_pct": round(self._daily_pnl / self.initial_capital * 100, 2),
            "trades_today": self._trades_today,
            "peak_equity": round(self._peak_equity, 2),
            "drawdown_pct": round((1 - (self._daily_start_equity / max(self._peak_equity, 1.0))) * 100, 2),
        }