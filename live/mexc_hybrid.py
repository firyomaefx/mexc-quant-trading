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
from datetime import datetime, timedelta
import time
import json
import urllib.request
import threading
import asyncio
import httpx

COINGECKO_IDS = {
    "XRP/USDT": "ripple",
    "ADA/USDT": "cardano",
    "SOL/USDT": "solana",
    "DOGE/USDT": "dogecoin",
    "LTC/USDT": "litecoin",
    "AVAX/USDT": "avalanche-2",
    "BTC/USDT": "bitcoin",
    "ETH/USDT": "ethereum",
    "BNB/USDT": "binancecoin",
    "DOT/USDT": "polkadot",
    "MATIC/USDT": "matic-network",
    "LINK/USDT": "chainlink",
    "TRX/USDT": "tron",
    "SHIB/USDT": "shiba-inu",
}


class MEXCHybridConnector:
    _CG_LAST_REQUEST = 0.0
    _CG_MIN_INTERVAL = 12.0
    _CG_LOCK = threading.Lock()

    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False,
                 proxy: str = None, use_coingecko: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.proxy = proxy or os.getenv("MEXC_PROXY", "")
        self.use_coingecko = use_coingecko
        self._connected = False
        self._mexc: Optional[ccxt.mexc] = None
        self._mexc_connected = False
        self._markets = {}
        self._last_request = 0.0
        self._ticker_cache: Dict[str, Dict] = {}
        self._ticker_cache_time = 0.0
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}
        self._ohlcv_cache_time: Dict[str, float] = {}
        self._http_client: Optional[httpx.Client] = None
        self._batch_cache: Dict[str, Dict] = {}
        self._batch_cache_time = 0.0

    def connect(self) -> bool:
        self._http_client = httpx.Client(timeout=15, follow_redirects=True)

        mexc_config = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        }

        if self.proxy:
            mexc_config["proxies"] = {
                "http": self.proxy,
                "https": self.proxy,
            }
            mexc_config["aiohttp_proxy"] = self.proxy
            mexc_config["aiohttp_trust_env"] = True

        self._mexc = ccxt.mexc(mexc_config)

        if self.proxy:
            self._mexc.session.proxies = {
                "http": self.proxy,
                "https": self.proxy,
            }

        try:
            self._mexc.load_markets()
            self._markets = self._mexc.markets
            self._mexc_connected = True
            self._connected = True
            print(f"MEXC direct connection: OK ({len(self._markets)} markets)")
            return True
        except Exception as e:
            print(f"MEXC direct connection failed: {e}")
            self._mexc_connected = False

        if self.use_coingecko:
            try:
                test = self._fetch_coingecko_prices(["ripple", "cardano", "solana"])
                if test:
                    self._connected = True
                    print(f"MEXC Hybrid: Using CoinGecko for market data (MEXC blocked)")
                    return True
            except Exception as e:
                print(f"CoinGecko also failed: {e}")

        print("All data sources failed. Check network connectivity.")
        return False

    def _coingecko_symbol(self, symbol: str) -> str:
        return COINGECKO_IDS.get(symbol, symbol.split("/")[0].lower())

    @classmethod
    def _cg_rate_limit(cls):
        with cls._CG_LOCK:
            now = time.time()
            elapsed = now - cls._CG_LAST_REQUEST
            if elapsed < cls._CG_MIN_INTERVAL:
                wait = cls._CG_MIN_INTERVAL - elapsed + 0.2
                time.sleep(wait)
            cls._CG_LAST_REQUEST = time.time()

    @classmethod
    def _cg_backoff(cls, seconds: float = 30.0):
        with cls._CG_LOCK:
            time.sleep(seconds)
            cls._CG_LAST_REQUEST = time.time()

    def _fetch_coingecko_prices(self, coin_ids: List[str]) -> Dict:
        self._cg_rate_limit()
        ids_str = ",".join(coin_ids)
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids_str}&vs_currencies=usd&include_24hr_change=true&include_24hr_vol=true"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantV2/2.0", "Accept": "application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._cg_backoff(30.0)
            return {}
        except Exception as e:
            print(f"CoinGecko price fetch failed: {e}")
            return {}

    def _fetch_coingecko_ohlcv(self, symbol: str, days: int = 1) -> pd.DataFrame:
        now = time.time()
        cache_time = self._ohlcv_cache_time.get(symbol, 0)
        if now - cache_time < 60 and symbol in self._ohlcv_cache:
            return self._ohlcv_cache[symbol]

        self._cg_rate_limit()
        coin_id = self._coingecko_symbol(symbol)
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantV2/2.0", "Accept": "application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                self._cg_backoff(30.0)
            if symbol in self._ohlcv_cache:
                return self._ohlcv_cache[symbol]
            return pd.DataFrame()
        except Exception as e:
            print(f"CoinGecko OHLCV fetch failed for {symbol}: {e}")
            if symbol in self._ohlcv_cache:
                return self._ohlcv_cache[symbol]
            return pd.DataFrame()

        prices = data.get("prices", [])
        volumes = data.get("total_volumes", [])

        if not prices:
            return pd.DataFrame()

        times = [pd.Timestamp(p[0], unit="ms") for p in prices]
        closes = [p[1] for p in prices]
        vol_map = {p[0]: p[1] for p in volumes}

        df = pd.DataFrame({
            "close": closes,
        }, index=times)
        df.index.name = "time"

        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0])
        df["high"] = df["close"].rolling(2, min_periods=1).max()
        df["low"] = df["close"].rolling(2, min_periods=1).min()

        df["volume"] = 0.0
        for i, row_time in enumerate(df.index):
            ts_ms = int(row_time.timestamp() * 1000)
            closest = min(vol_map.keys(), key=lambda x: abs(x - ts_ms)) if vol_map else 0
            if closest:
                df.iloc[i, df.columns.get_loc("volume")] = vol_map.get(closest, 0)

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)

        self._ohlcv_cache[symbol] = df
        self._ohlcv_cache_time[symbol] = time.time()
        return df

    def _fetch_coingecko_markets(self, symbols: List[str]) -> Dict[str, Dict]:
        now = time.time()
        if now - self._batch_cache_time < 30 and self._batch_cache:
            result = {}
            for sym in symbols:
                if sym in self._batch_cache:
                    result[sym] = self._batch_cache[sym]
            if result:
                return result

        self._cg_rate_limit()
        coin_ids = ",".join([self._coingecko_symbol(s) for s in symbols])
        url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={coin_ids}&order=market_cap_desc&per_page=50&sparkline=false"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantV2/2.0", "Accept": "application/json"})
            resp = urllib.request.urlopen(req, timeout=15)
            data = json.loads(resp.read().decode())
            result = {}
            for d in data:
                for sym in symbols:
                    if self._coingecko_symbol(sym) == d["id"]:
                        result[sym] = {
                            "last": d.get("current_price", 0),
                            "bid": d.get("current_price", 0) * (1 - 0.0003),
                            "ask": d.get("current_price", 0) * (1 + 0.0003),
                            "high_24h": d.get("high_24h", 0),
                            "low_24h": d.get("low_24h", 0),
                            "volume_24h": d.get("total_volume", 0),
                            "change_pct": d.get("price_change_percentage_24h", 0),
                            "market_cap": d.get("market_cap", 0),
                        }
                        self._batch_cache[sym] = result[sym]
            self._batch_cache_time = now
            return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                print(f"CoinGecko rate limit hit, backing off 30s...")
                self._cg_backoff(30.0)
            else:
                print(f"CoinGecko markets fetch failed: {e}")
            return {}
        except Exception as e:
            print(f"CoinGecko markets fetch failed: {e}")
            return {}

    def _rate_limit(self):
        now = time.time()
        elapsed = now - self._last_request
        if elapsed < 0.06:
            time.sleep(0.06 - elapsed)
        self._last_request = time.time()

    def fetch_rates(self, symbol: str, timeframe: str = "1m", limit: int = 500,
                    since: Optional[int] = None) -> pd.DataFrame:
        self._rate_limit()

        if self._mexc_connected:
            try:
                return self._fetch_mexc_ohlcv(symbol, timeframe, limit, since)
            except Exception as e:
                print(f"MEXC OHLCV failed, falling back to CoinGecko: {e}")

        days = max(1, limit // 1440 + 1)
        df = self._fetch_coingecko_ohlcv(symbol, days=days)
        if df.empty:
            raise RuntimeError(f"No OHLCV data available for {symbol}")
        self._ohlcv_cache[symbol] = df
        return df

    def _fetch_mexc_ohlcv(self, symbol: str, timeframe: str = "1m",
                          limit: int = 500, since: Optional[int] = None) -> pd.DataFrame:
        ohlcv = self._mexc.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        if not ohlcv:
            raise RuntimeError(f"No OHLCV data from MEXC for {symbol}")
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df.set_index("time", inplace=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(np.float64)
        return df

    def get_current_price(self, symbol: str) -> Tuple[float, float]:
        self._rate_limit()

        if self._mexc_connected:
            try:
                return self._get_mexc_price(symbol)
            except Exception:
                pass

        tickers = self._batch_fetch_tickers([symbol])
        if symbol in tickers:
            t = tickers[symbol]
            bid = t.get("bid", 0)
            ask = t.get("ask", 0)
            if bid > 0 and ask > 0:
                return bid, ask
            last = t.get("last", 0)
            if last > 0:
                spread_pct = 0.0006
                return last * (1 - spread_pct / 2), last * (1 + spread_pct / 2)

        return 0.0, 0.0

    def _batch_fetch_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        markets = self._fetch_coingecko_markets(symbols)
        found = {s: markets[s] for s in symbols if s in markets}
        missing = [s for s in symbols if s not in markets]
        if missing:
            prices = self._fetch_coingecko_prices([self._coingecko_symbol(s) for s in missing])
            for s in missing:
                coin_id = self._coingecko_symbol(s)
                if coin_id in prices:
                    p = prices[coin_id]
                    usd = p.get("usd", 0)
                    found[s] = {
                        "symbol": s, "bid": usd * 0.9997, "ask": usd * 1.0003,
                        "last": usd, "high_24h": usd, "low_24h": usd,
                        "volume_24h": p.get("usd_24h_vol", 0),
                        "change_pct": p.get("usd_24h_change", 0),
                    }
        return found

    def _get_mexc_price(self, symbol: str) -> Tuple[float, float]:
        ticker = self._mexc.fetch_ticker(symbol)
        bid = ticker.get("bid", 0) or ticker.get("last", 0)
        ask = ticker.get("ask", 0) or ticker.get("last", 0)
        return bid, ask

    def get_ticker(self, symbol: str) -> Dict:
        self._rate_limit()

        if self._mexc_connected:
            try:
                return self._get_mexc_ticker(symbol)
            except Exception:
                pass

        cache_key = f"ticker_{symbol}"
        now = time.time()
        if now - self._ticker_cache_time < 15 and cache_key in self._ticker_cache:
            return self._ticker_cache[cache_key]

        tickers = self._batch_fetch_tickers([symbol])
        if symbol in tickers:
            self._ticker_cache[cache_key] = tickers[symbol]
            self._ticker_cache_time = now
            return tickers[symbol]

        return {"symbol": symbol, "bid": 0, "ask": 0, "last": 0, "high_24h": 0, "low_24h": 0, "volume_24h": 0, "change_pct": 0}

    def _get_mexc_ticker(self, symbol: str) -> Dict:
        t = self._mexc.fetch_ticker(symbol)
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

    def get_spread(self, symbol: str) -> float:
        bid, ask = self.get_current_price(symbol)
        if bid <= 0 or ask <= 0:
            return 0.0006
        return (ask - bid) / bid

    def get_spread_pct(self, symbol: str) -> float:
        return self.get_spread(symbol) * 100.0

    def get_balance(self, quote: str = "USDT") -> Dict:
        if self._mexc_connected:
            try:
                balance = self._mexc.fetch_balance()
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
                print(f"MEXC balance fetch failed: {e}")

        print("Balance query requires MEXC connection. Using config capital.")
        return {
            "quote": quote, "balance": 0.0, "available": 0.0,
            "currency": quote, "equity": 0.0, "margin": 0.0,
            "free_margin": 0.0, "leverage": 1,
        }

    def place_order(self, symbol: str, order_type: str, amount: float,
                    price: float = 0.0, params: Dict = None) -> Optional[Dict]:
        if self._mexc_connected:
            try:
                return self._place_mexc_order(symbol, order_type, amount, price, params)
            except Exception as e:
                print(f"MEXC order failed: {e}")
                return None

        print(f"ORDER REJECTED: MEXC API not connected. Cannot place real order for {symbol} {order_type} {amount}")
        return None

    def _place_mexc_order(self, symbol: str, order_type: str, amount: float,
                           price: float = 0.0, params: Dict = None) -> Optional[Dict]:
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
            order = self._mexc.create_market_buy_order(symbol, amount, params)
        elif order_type.upper() == "SELL":
            order = self._mexc.create_market_sell_order(symbol, amount, params)
        elif order_type.upper() == "LIMIT_BUY":
            order = self._mexc.create_limit_buy_order(symbol, amount, price, params)
        elif order_type.upper() == "LIMIT_SELL":
            order = self._mexc.create_limit_sell_order(symbol, amount, price, params)
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

    def close_position_spot(self, symbol: str, amount: float = None) -> bool:
        if not self._mexc_connected:
            print("Cannot close position: MEXC not connected")
            return False

        base = symbol.split("/")[0]
        if amount is None:
            try:
                bal = self._mexc.fetch_balance()
                amount = float(bal.get("free", {}).get(base, 0))
            except Exception:
                pass

        if amount is None or amount <= 0:
            return False

        result = self.place_order(symbol, "SELL", amount)
        return result is not None and result.get("status") in ("closed", "ok", "filled")

    def close_position(self, symbol: str, amount: float = None) -> bool:
        if not self._mexc_connected:
            print("Cannot close position: MEXC not connected")
            return False

        base = symbol.split("/")[0]
        if amount is None:
            try:
                bal = self._mexc.fetch_balance()
                amount = float(bal.get("free", {}).get(base, 0))
            except Exception:
                pass

        if amount is None or amount <= 0:
            return False

        order_type = "SELL"
        result = self.place_order(symbol, order_type, amount)
        return result is not None and result.get("status") in ("closed", "ok", "filled")

    def get_positions(self) -> List[Dict]:
        if self._mexc_connected:
            try:
                total = self._mexc.fetch_balance().get("total", {})
                positions = []
                for asset, amt in total.items():
                    if asset in ("USDT", "USDC", "BUSD") or amt <= 0:
                        continue
                    symbol = f"{asset}/USDT"
                    if symbol in self._markets:
                        bid, ask = self.get_current_price(symbol)
                        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
                        positions.append({
                            "symbol": symbol, "side": "buy", "amount": float(amt),
                            "entry_price": 0.0, "current_price": mid,
                            "unrealized_pnl": 0.0, "quote_value": float(amt) * mid if mid > 0 else 0.0,
                        })
                return positions
            except Exception:
                pass
        return []

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        if self._mexc_connected:
            try:
                orders = self._mexc.fetch_open_orders(symbol)
                return [{"id": o.get("id"), "symbol": o.get("symbol"), "type": o.get("side"),
                         "amount": o.get("amount"), "price": o.get("price", 0),
                         "status": o.get("status"), "timestamp": o.get("timestamp")} for o in orders]
            except Exception:
                pass
        return []

    def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        if not self._mexc_connected:
            return 0
        orders = self.get_open_orders(symbol)
        cancelled = 0
        for o in orders:
            try:
                self._mexc.cancel_order(o["id"], o["symbol"])
                cancelled += 1
            except Exception:
                pass
        return cancelled

    def get_symbol_info(self, symbol: str) -> Dict:
        if symbol in self._markets:
            market = self._markets[symbol]
            limits = market.get("limits", {})
            precision = market.get("precision", {})
            return {
                "symbol": symbol, "base": market.get("base", ""), "quote": market.get("quote", ""),
                "min_amount": limits.get("amount", {}).get("min", 0),
                "max_amount": limits.get("amount", {}).get("max", None),
                "min_cost": limits.get("cost", {}).get("min", 0),
                "price_precision": precision.get("price", 8),
                "amount_precision": precision.get("amount", 8),
                "taker_fee": market.get("taker", 0.002),
                "maker_fee": market.get("maker", 0.002),
            }

        defaults = {
            "XRP/USDT": {"min_amount": 1, "min_cost": 1, "price_precision": 6, "amount_precision": 1},
            "ADA/USDT": {"min_amount": 1, "min_cost": 1, "price_precision": 5, "amount_precision": 1},
            "SOL/USDT": {"min_amount": 0.01, "min_cost": 1, "price_precision": 4, "amount_precision": 2},
        }
        d = defaults.get(symbol, {"min_amount": 0.001, "min_cost": 1, "price_precision": 8, "amount_precision": 8})
        return {
            "symbol": symbol, "base": symbol.split("/")[0], "quote": "USDT",
            "min_amount": d["min_amount"], "max_amount": None,
            "min_cost": d["min_cost"], "price_precision": d["price_precision"],
            "amount_precision": d["amount_precision"], "taker_fee": 0.001, "maker_fee": 0.001,
        }

    def place_sl_tp(self, symbol: str, side: str, amount: float,
                    stop_price: float, take_profit: float) -> Optional[Dict]:
        if self._mexc_connected:
            try:
                params = {"stopPrice": stop_price, "takeProfitPrice": take_profit}
                if side.upper() == "BUY":
                    return self._place_mexc_order(symbol, "LIMIT_BUY", amount, stop_price, params)
                else:
                    return self._place_mexc_order(symbol, "LIMIT_SELL", amount, take_profit, params)
            except Exception as e:
                print(f"SL/TP placement failed: {e}")
        return None

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        if self._mexc_connected:
            try:
                self._mexc.set_leverage(leverage, symbol)
                return True
            except Exception:
                pass
        return False

    def set_margin_mode(self, symbol: str, mode: str = "cross") -> bool:
        return False

    def disconnect(self):
        if self._mexc:
            self._mexc = None
        if self._http_client:
            self._http_client.close()
            self._http_client = None
        self._mexc_connected = False
        self._connected = False
        print("MEXC Hybrid disconnected.")

    @property
    def connected(self) -> bool:
        return self._connected

    @connected.setter
    def connected(self, value: bool):
        self._connected = value

    @property
    def mexc_connected(self) -> bool:
        return self._mexc_connected