import sys
import os
_script = os.path.dirname(os.path.abspath(__file__))
_grandparent = os.path.dirname(os.path.dirname(_script))
if _grandparent not in sys.path:
    sys.path.insert(0, _grandparent)

import time
import threading
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from datetime import datetime


class DashboardDataProvider:
    def __init__(self, config, scalper=None, paper_trader=None):
        self.config = config
        self.scalper = scalper
        self.paper_trader = paper_trader
        self.pairs = config.scalping.pairs
        self.connected = False
        self._last_refresh = 0

        self._equity_curve: List[float] = [config.scalping.initial_capital]
        self._trade_log: List[Dict] = []
        self._pair_data: Dict[str, Dict] = {p: self._empty_pair() for p in self.pairs}
        self._account = {
            "equity": config.scalping.initial_capital,
            "balance": config.scalping.initial_capital,
            "daily_pnl": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "expectancy": 0.0,
            "drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "consecutive_losses": 0,
            "breaker_status": "OK",
            "is_halted": False,
            "sentiment": 0.0,
        }

    def _empty_pair(self) -> Dict:
        return {
            "price": 0.0, "signal": 0, "zscore": 0.0,
            "hurst": 0.5, "ml_conf": 0.0, "spread": 0.0,
            "volume_ratio": 1.0, "mtf_conf": 1.0, "rl_conf": 1.0,
        }

    def connect(self) -> bool:
        if self.scalper:
            try:
                connected = self.scalper.mexc.connected if hasattr(self.scalper.mexc, 'connected') else False
                self.connected = connected
                return connected
            except Exception:
                self.connected = False
                return False
        elif self.paper_trader:
            self.connected = True
            return True
        return False

    def refresh(self) -> Dict:
        if self.scalper:
            self._refresh_from_scalper()
        elif self.paper_trader:
            self._refresh_from_paper()

        self._account["equity_curve"] = list(self._equity_curve[-500:])
        self._account["trade_log"] = list(self._trade_log)
        self._account["pair_data"] = dict(self._pair_data)

        return self._account

    def _refresh_from_scalper(self):
        try:
            if not self.connected:
                try:
                    self.connected = self.scalper.mexc.connect()
                except Exception:
                    pass

            if self.connected:
                for p in self.pairs:
                    try:
                        ticker = self.scalper.mexc.get_ticker(p)
                        self._pair_data[p]["price"] = ticker.get("last", 0)
                    except Exception:
                        pass

            for sym, pos in self.scalper._open_positions.items():
                pid = sym.replace("/", "")
                if pid in [p.replace("/", "") for p in self.pairs]:
                    self._pair_data[sym]["signal"] = pos.get("direction", 0)

            for sym in self.scalper.signal_gens:
                features = self.scalper._features_cache.get(sym)
                if features is not None and len(features) > 0:
                    latest = features.iloc[-1]
                    self._pair_data[sym]["zscore"] = float(latest.get("zscore", 0))
                    self._pair_data[sym]["hurst"] = float(latest.get("hurst", 0.5))
                    self._pair_data[sym]["signal"] = int(latest.get("signal", 0))

            cb = self.scalper.circuit_breaker.get_status()
            self._account["is_halted"] = cb.get("is_halted", False)
            self._account["breaker_status"] = cb.get("halt_reason", "OK")
            self._account["daily_pnl"] = cb.get("daily_pnl", 0)
            self._account["consecutive_losses"] = cb.get("consecutive_losses", 0)
            self._account["daily_loss_pct"] = abs(cb.get("daily_pnl", 0)) / self.config.scalping.initial_capital * 100

            trades = list(self.scalper._trade_history)
            if trades:
                wins = [t for t in trades if t.get("pnl", 0) > 0]
                self._account["win_rate"] = len(wins) / len(trades) * 100
                self._account["total_trades"] = len(trades)
                pnls = [t.get("pnl", 0) for t in trades]
                self._account["expectancy"] = np.mean(pnls) if pnls else 0
                self._account["daily_pnl"] = sum(pnls)

                last_equity = self.config.scalping.initial_capital + sum(pnls)
                self._equity_curve.append(last_equity)
                if len(self._equity_curve) > 500:
                    self._equity_curve = self._equity_curve[-500:]

                self._trade_log = trades[-50:]

            try:
                sentiment = self.scalper.sentiment.get_sentiment()
                self._account["sentiment"] = sentiment
            except Exception:
                pass

            try:
                balance = self.scalper.mexc.get_balance("USDT")
                self._account["equity"] = balance.get("equity", self.config.scalping.initial_capital)
            except Exception:
                pass

        except Exception:
            pass

    def _refresh_from_paper(self):
        try:
            pt = self.paper_trader
            self._account["equity"] = pt._equity
            self._account["daily_pnl"] = pt._daily_pnl
            self._account["daily_loss_pct"] = abs(pt._daily_pnl) / self.config.scalping.initial_capital * 100

            for sym, df in pt._data_cache.items():
                if len(df) > 0:
                    self._pair_data[sym]["price"] = float(df["close"].iloc[-1])

            for sym, features in pt._features_cache.items():
                if features is not None and len(features) > 0:
                    latest = features.iloc[-1]
                    self._pair_data[sym]["zscore"] = float(latest.get("zscore", 0))
                    self._pair_data[sym]["hurst"] = float(latest.get("hurst", 0.5))
                    self._pair_data[sym]["signal"] = int(latest.get("signal", 0))

            cb = pt.circuit_breaker.get_status()
            self._account["is_halted"] = cb.get("is_halted", False)
            self._account["breaker_status"] = cb.get("halt_reason", "OK")
            self._account["consecutive_losses"] = cb.get("consecutive_losses", 0)

            trades = list(pt._trade_history)
            if trades:
                wins = [t for t in trades if t.get("pnl", 0) > 0]
                self._account["win_rate"] = len(wins) / len(trades) * 100
                self._account["total_trades"] = len(trades)
                pnls = [t.get("pnl", 0) for t in trades]
                self._account["expectancy"] = np.mean(pnls) if pnls else 0
                self._trade_log = trades[-50:]

            self._equity_curve.append(pt._equity)
            if len(self._equity_curve) > 500:
                self._equity_curve = self._equity_curve[-500:]

            try:
                sentiment = pt.sentiment.get_sentiment()
                self._account["sentiment"] = sentiment
            except Exception:
                pass

        except Exception:
            pass