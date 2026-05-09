import sys

import ccxt
import time
from typing import Optional, List, Dict
from datetime import datetime

from live.mexc_adapter import MEXCConnector


class MEXCFuturesConnector(MEXCConnector):
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        super().__init__(api_key, api_secret, testnet)
        self._leverage: Dict[str, int] = {}
        self._margin_mode: Dict[str, str] = {}

    def connect(self) -> bool:
        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},
        }
        try:
            self.exchange = ccxt.mexc(config)
            self.exchange.load_markets()
            self._markets = self.exchange.markets
            self.connected = True
            swap_count = sum(1 for m in self._markets.values() if m.get("swap"))
            print(f"MEXC Futures connected. {swap_count} swap markets loaded.")
            return True
        except Exception as e:
            print(f"MEXC Futures connection failed: {e}")
            return False

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        self._rate_limit()
        try:
            self.exchange.set_leverage(leverage, symbol)
            self._leverage[symbol] = leverage
            print(f"{symbol} leverage set to {leverage}x")
            return True
        except Exception as e:
            print(f"Failed to set leverage for {symbol}: {e}")
            return False

    def set_margin_mode(self, symbol: str, mode: str = "isolated") -> bool:
        self._rate_limit()
        try:
            self.exchange.set_margin_mode(mode, symbol)
            self._margin_mode[symbol] = mode
            print(f"{symbol} margin mode set to {mode}")
            return True
        except Exception as e:
            print(f"Failed to set margin mode for {symbol}: {e}")
            return False

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        self._rate_limit()
        try:
            symbols = [symbol] if symbol else None
            positions = self.exchange.fetch_positions(symbols)
            result = []
            for pos in positions:
                contracts = float(pos.get("contracts", 0))
                if contracts == 0:
                    continue
                result.append({
                    "symbol": pos.get("symbol", ""),
                    "side": pos.get("side", "long"),
                    "amount": contracts,
                    "entry_price": float(pos.get("entryPrice", 0)),
                    "current_price": float(pos.get("markPrice", 0)),
                    "unrealized_pnl": float(pos.get("unrealizedPnl", 0)),
                    "liquidation_price": float(pos.get("liquidationPrice", 0)),
                    "leverage": int(pos.get("leverage", 1)),
                    "margin_mode": pos.get("marginMode", "isolated"),
                    "initial_margin": float(pos.get("initialMargin", 0)),
                    "maintenance_margin": float(pos.get("maintenanceMargin", 0)),
                    "percentage": float(pos.get("percentage", 0)),
                })
            return result
        except Exception as e:
            print(f"Failed to fetch futures positions: {e}")
            return []

    def get_liquidation_price(self, symbol: str, side: str = "long") -> float:
        positions = self.get_positions(symbol)
        for pos in positions:
            if pos["symbol"] == symbol and pos["side"] == side:
                return pos["liquidation_price"]
        return 0.0

    def get_funding_rate(self, symbol: str) -> float:
        self._rate_limit()
        try:
            funding = self.exchange.fetch_funding_rate(symbol)
            return float(funding.get("fundingRate", 0))
        except Exception:
            return 0.0

    def get_funding_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        self._rate_limit()
        try:
            return self.exchange.fetch_funding_rate_history(symbol, limit=limit)
        except Exception:
            return []

    def place_order(self, symbol: str, order_type: str, amount: float,
                    price: float = 0.0, params: Dict = None) -> Optional[Dict]:
        if params is None:
            params = {}

        if "reduceOnly" not in params:
            params["reduceOnly"] = False

        return super().place_order(symbol, order_type, amount, price, params)

    def place_sl_tp(self, symbol: str, side: str, amount: float,
                    stop_loss: float = 0.0, take_profit: float = 0.0) -> Optional[Dict]:
        self._rate_limit()
        try:
            params = {}
            is_long = side.upper() in ("BUY", "LONG")

            if stop_loss > 0:
                sl_type = "STOP_MARKET"
                sl_side = "SELL" if is_long else "BUY"
                sl_price = stop_loss
            else:
                sl_price = 0

            if take_profit > 0:
                tp_type = "TAKE_PROFIT_MARKET"
                tp_side = "SELL" if is_long else "BUY"
                tp_price = take_profit
            else:
                tp_price = 0

            if sl_price > 0:
                self.exchange.create_order(
                    symbol, "stop_market", sl_side, amount, sl_price,
                    {"stopLossPrice": sl_price, "reduceOnly": True}
                )
            if tp_price > 0:
                self.exchange.create_order(
                    symbol, "take_profit_market", tp_side, amount, tp_price,
                    {"takeProfitPrice": tp_price, "reduceOnly": True}
                )

            return {"status": "ok"}
        except Exception as e:
            print(f"SL/TP placement failed for {symbol}: {e}")
            return None

    def close_position(self, symbol: str, amount: float = None) -> bool:
        positions = self.get_positions(symbol)
        for pos in positions:
            if pos["symbol"] == symbol:
                close_amount = amount or pos["amount"]
                close_side = "SELL" if pos["side"] == "long" else "BUY"
                result = self.place_order(symbol, close_side, close_amount,
                                          params={"reduceOnly": True})
                return result is not None
        return False

    def close_all_positions(self) -> int:
        positions = self.get_positions()
        closed = 0
        for pos in positions:
            if self.close_position(pos["symbol"]):
                closed += 1
        return closed

    def get_balance(self, quote: str = "USDT") -> Dict:
        self._rate_limit()
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get("total", {})
            free = balance.get("free", {})
            used = balance.get("used", {})
            info = balance.get("info", {})
            usdt_balance = float(total.get(quote, 0))
            available = float(free.get(quote, 0))
            margin_used = float(used.get(quote, 0))
            return {
                "quote": quote,
                "balance": usdt_balance,
                "available": available,
                "currency": quote,
                "equity": usdt_balance,
                "margin": margin_used,
                "free_margin": available,
                "leverage": max(self._leverage.values()) if self._leverage else 1,
            }
        except Exception as e:
            print(f"Futures balance fetch failed: {e}")
            return {
                "quote": quote, "balance": 0.0, "available": 0.0,
                "currency": quote, "equity": 0.0, "margin": 0.0,
                "free_margin": 0.0, "leverage": 1,
            }

    def get_position_risk(self, symbol: str) -> Dict:
        positions = self.get_positions(symbol)
        for pos in positions:
            if pos["symbol"] == symbol:
                liq_price = pos.get("liquidation_price", 0)
                current = pos.get("current_price", 0)
                if current > 0 and liq_price > 0:
                    if pos["side"] == "long":
                        distance_pct = (current - liq_price) / current
                    else:
                        distance_pct = (liq_price - current) / current
                    pos["liquidation_distance_pct"] = max(0, distance_pct)
                else:
                    pos["liquidation_distance_pct"] = 1.0
                return pos
        return {"liquidation_distance_pct": 1.0}
