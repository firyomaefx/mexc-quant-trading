import sys
import os
_quant_v2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_quant_v2)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import ccxt
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import time
import hmac
import hashlib


class MEXCConnector:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.exchange: Optional[ccxt.mexc] = None
        self.connected = False
        self._markets = {}
        self._last_request = 0.0

    def connect(self) -> bool:
        config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }
        if self.testnet:
            config["options"]["defaultType"] = "spot"
            self.exchange = ccxt.mexc(config)
            self.exchange.urls["api"] = self.exchange.urls.get("test", self.exchange.urls["api"])
        else:
            self.exchange = ccxt.mexc(config)

        try:
            self.exchange.load_markets()
            self._markets = self.exchange.markets
            self.connected = True
            print(f"MEXC Spot connected. {len(self._markets)} markets loaded.")
            return True
        except Exception as e:
            print(f"MEXC connection failed: {e}")
            return False

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < 0.05:
            time.sleep(0.05 - elapsed)
        self._last_request = time.time()

    def fetch_rates(self, symbol: str, timeframe: str = "1m", limit: int = 500,
                    since: Optional[int] = None) -> pd.DataFrame:
        self._rate_limit()
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch {symbol} {timeframe}: {e}")

        if not ohlcv:
            raise RuntimeError(f"No OHLCV data for {symbol} {timeframe}")

        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)
        return df

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        self._rate_limit()
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker.get("bid", 0.0) or ticker.get("last", 0.0), \
                   ticker.get("ask", 0.0) or ticker.get("last", 0.0)
        except Exception as e:
            raise RuntimeError(f"Cannot get price for {symbol}: {e}")

    def get_spread(self, symbol: str) -> float:
        bid, ask = self.get_current_price(symbol)
        if bid <= 0 or ask <= 0:
            return 0.0
        return (ask - bid) / bid

    def get_spread_pct(self, symbol: str) -> float:
        return self.get_spread(symbol) * 100.0

    def get_orderbook(self, symbol: str, depth: int = 10) -> Dict:
        self._rate_limit()
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=depth)
            return {
                "bids": ob.get("bids", [])[:depth],
                "asks": ob.get("asks", [])[:depth],
                "timestamp": ob.get("timestamp", 0),
            }
        except Exception as e:
            print(f"Orderbook fetch failed for {symbol}: {e}")
            return {"bids": [], "asks": [], "timestamp": 0}

    def get_balance(self, quote: str = "USDT") -> Dict:
        self._rate_limit()
        try:
            balance = self.exchange.fetch_balance()
            total = balance.get("total", {})
            free = balance.get("free", {})
            return {
                "quote": quote,
                "balance": float(total.get(quote, 0)),
                "available": float(free.get(quote, 0)),
                "currency": quote,
                "equity": float(total.get(quote, 0)),
                "margin": 0.0,
                "free_margin": float(free.get(quote, 0)),
                "leverage": 1,
            }
        except Exception as e:
            print(f"Balance fetch failed: {e}")
            return {
                "quote": quote, "balance": 0.0, "available": 0.0,
                "currency": quote, "equity": 0.0, "margin": 0.0,
                "free_margin": 0.0, "leverage": 1,
            }

    def place_order(self, symbol: str, order_type: str, amount: float,
                    price: float = 0.0, params: Dict = None) -> Optional[Dict]:
        self._rate_limit()
        try:
            if params is None:
                params = {}

            market = self._markets.get(symbol, {})
            precision = market.get("precision", {})
            amount_precision = precision.get("amount", 8)
            amount = round(amount, amount_precision)

            min_amount = market.get("limits", {}).get("amount", {}).get("min", 0)
            if amount < min_amount:
                print(f"Amount {amount} below min {min_amount} for {symbol}")
                return None

            if order_type.upper() == "BUY":
                order = self.exchange.create_market_buy_order(symbol, amount, params)
            elif order_type.upper() == "SELL":
                order = self.exchange.create_market_sell_order(symbol, amount, params)
            elif order_type.upper() == "LIMIT_BUY":
                order = self.exchange.create_limit_buy_order(symbol, amount, price, params)
            elif order_type.upper() == "LIMIT_SELL":
                order = self.exchange.create_limit_sell_order(symbol, amount, price, params)
            else:
                raise ValueError(f"Invalid order_type: {order_type}")

            return {
                "id": order.get("id"),
                "symbol": order.get("symbol", symbol),
                "type": order_type,
                "amount": order.get("amount", amount),
                "price": order.get("price", order.get("average", price)),
                "cost": order.get("cost", 0),
                "status": order.get("status", "unknown"),
                "filled": order.get("filled", amount),
                "timestamp": order.get("timestamp", int(time.time() * 1000)),
            }
        except Exception as e:
            print(f"Order failed for {symbol} ({order_type} {amount}): {e}")
            return None

    def close_position_spot(self, symbol: str, amount: float = None) -> bool:
        balances = self.get_balance()
        base = symbol.split("/")[0]
        if amount is None:
            try:
                bal = self.exchange.fetch_balance()
                amount = float(bal.get("free", {}).get(base, 0))
            except Exception:
                pass

        if amount is None or amount <= 0:
            return False

        result = self.place_order(symbol, "SELL", amount)
        return result is not None and result.get("status") in ("closed", "ok", "filled")

    def get_positions(self) -> List[Dict]:
        balances = self.get_balance()
        total = self.exchange.fetch_balance().get("total", {})
        positions = []
        for asset, amount in total.items():
            if asset in ("USDT", "USDC", "BUSD") or amount <= 0:
                continue
            symbol = f"{asset}/USDT"
            if symbol in self._markets:
                bid, ask = self.get_current_price(symbol)
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                positions.append({
                    "symbol": symbol,
                    "side": "buy",
                    "amount": float(amount),
                    "entry_price": 0.0,
                    "current_price": mid,
                    "unrealized_pnl": 0.0,
                    "quote_value": float(amount) * mid if mid > 0 else 0.0,
                })
        return positions

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        self._rate_limit()
        try:
            orders = self.exchange.fetch_open_orders(symbol)
            return [
                {
                    "id": o.get("id"),
                    "symbol": o.get("symbol"),
                    "type": o.get("side"),
                    "amount": o.get("amount"),
                    "price": o.get("price", 0),
                    "status": o.get("status"),
                    "timestamp": o.get("timestamp"),
                }
                for o in orders
            ]
        except Exception as e:
            print(f"Failed to fetch open orders: {e}")
            return []

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        if symbol:
            orders = self.get_open_orders(symbol)
        else:
            orders = self.get_open_orders()
        cancelled = 0
        for o in orders:
            try:
                self.exchange.cancel_order(o["id"], o["symbol"])
                cancelled += 1
            except Exception:
                pass
        return cancelled

    def get_symbol_info(self, symbol: str) -> Dict:
        market = self._markets.get(symbol, {})
        limits = market.get("limits", {})
        precision = market.get("precision", {})
        return {
            "symbol": symbol,
            "base": market.get("base", ""),
            "quote": market.get("quote", ""),
            "min_amount": limits.get("amount", {}).get("min", 0),
            "max_amount": limits.get("amount", {}).get("max", None),
            "min_cost": limits.get("cost", {}).get("min", 0),
            "price_precision": precision.get("price", 8),
            "amount_precision": precision.get("amount", 8),
            "taker_fee": market.get("taker", 0.002),
            "maker_fee": market.get("maker", 0.002),
        }

    def get_ticker(self, symbol: str) -> Dict:
        self._rate_limit()
        try:
            t = self.exchange.fetch_ticker(symbol)
            return {
                "symbol": symbol,
                "bid": t.get("bid", 0),
                "ask": t.get("ask", 0),
                "last": t.get("last", 0),
                "high_24h": t.get("high", 0),
                "low_24h": t.get("low", 0),
                "volume_24h": t.get("baseVolume", 0),
                "change_pct": t.get("percentage", 0),
            }
        except Exception as e:
            print(f"Ticker fetch failed for {symbol}: {e}")
            return {}

    def disconnect(self):
        if self.exchange:
            self.exchange = None
        self.connected = False
        print("MEXC Spot disconnected.")
