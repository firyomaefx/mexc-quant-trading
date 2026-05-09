import sys
import os
_quant_v2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_quant_v2)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import asyncio
import json
import time
import zlib
from typing import Optional, Dict, List, Callable, Set
from datetime import datetime
from collections import deque
import threading
import websockets


class MEXCWebSocket:
    SPOT_WS_URL = "wss://wbs.mexc.com/ws"
    FUTURES_WS_URL = "wss://contract.mexc.com/edge"

    def __init__(self, market_type: str = "spot", max_cached_candles: int = 1000):
        self.market_type = market_type
        self.ws_url = self.SPOT_WS_URL if market_type == "spot" else self.FUTURES_WS_URL
        self._ws = None
        self._running = False
        self._loop = None
        self._thread = None
        self._subscribed_streams: Set[str] = set()
        self._callbacks: Dict[str, List[Callable]] = {}

        self._candles: Dict[str, deque] = {}
        self._tickers: Dict[str, Dict] = {}
        self._orderbooks: Dict[str, Dict] = {}
        self._max_candles = max_cached_candles

        self._lock = threading.Lock()

    def on(self, event: str, callback: Callable):
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, *args):
        for cb in self._callbacks.get(event, []):
            try:
                cb(*args)
            except Exception:
                pass

    def get_candles(self, symbol: str, timeframe: str = "1m") -> deque:
        key = f"{symbol}_{timeframe}".lower().replace("/", "")
        with self._lock:
            if key not in self._candles:
                self._candles[key] = deque(maxlen=self._max_candles)
            return self._candles[key]

    def get_ticker(self, symbol: str) -> Dict:
        with self._lock:
            return self._tickers.get(symbol, {})

    def get_orderbook(self, symbol: str) -> Dict:
        with self._lock:
            return self._orderbooks.get(symbol, {})

    def subscribe_kline(self, symbol: str, timeframe: str = "Min1"):
        if self.market_type == "spot":
            stream = f"spot@public.kline.v3.api@{symbol}@{timeframe}"
        else:
            stream = f"sub.kline@{symbol}@{timeframe}"
        self._subscribe(stream)

    def subscribe_ticker(self, symbol: str):
        if self.market_type == "spot":
            stream = f"spot@public.bookTicker.v3.api@{symbol}"
        else:
            stream = f"sub.ticker@{symbol}"
        self._subscribe(stream)

    def subscribe_depth(self, symbol: str, depth: int = 20):
        if self.market_type == "spot":
            stream = f"spot@public.increase.depth.v3.api@{symbol}"
        else:
            stream = f"sub.depth@{symbol}@{depth}"
        self._subscribe(stream)

    def subscribe_mini_ticker(self, symbol: str):
        if self.market_type == "spot":
            stream = f"spot@public.miniTicker.v3.api@{symbol}"
        else:
            stream = f"sub.miniTicker@{symbol}"
        self._subscribe(stream)

    def _subscribe(self, stream: str):
        with self._lock:
            self._subscribed_streams.add(stream)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        time.sleep(0.5)

    def stop(self):
        self._running = False

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect_and_listen())

    async def _connect_and_listen(self):
        while self._running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws

                    streams = list(self._subscribed_streams)
                    for i in range(0, len(streams), 10):
                        batch = streams[i:i + 10]
                        sub_msg = self._build_subscription(batch)
                        if sub_msg:
                            await ws.send(json.dumps(sub_msg))
                        await asyncio.sleep(0.2)

                    self._emit("connected")
                    await self._read_loop(ws)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._emit("error", str(e))
                if self._running:
                    await asyncio.sleep(3)

    def _build_subscription(self, streams: List[str]) -> Optional[Dict]:
        if not streams:
            return None
        if self.market_type == "spot":
            return {"method": "SUBSCRIPTION", "params": streams, "id": int(time.time() * 1000)}
        else:
            return {"method": "sub", "params": streams, "id": int(time.time())}

    async def _read_loop(self, ws):
        while self._running:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=60)
                self._process_message(msg)
            except asyncio.TimeoutError:
                await ws.ping()
            except websockets.ConnectionClosed:
                break

    def _process_message(self, raw: str):
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return

        if self.market_type == "spot":
            self._process_spot_message(data)
        else:
            self._process_futures_message(data)

    def _process_spot_message(self, data: Dict):
        channel = data.get("c", "")
        d = data.get("d", {})
        symbol = data.get("s", d.get("s", ""))
        event_type = data.get("e", channel)

        if "kline" in channel or event_type == "kline":
            self._handle_spot_kline(symbol, d, data)
        elif "bookTicker" in channel or event_type == "bookTicker":
            self._handle_spot_ticker(symbol, d, data)
        elif "depth" in channel or event_type == "depth":
            self._handle_spot_depth(symbol, d, data)
        elif "miniTicker" in channel or event_type == "miniTicker":
            self._handle_spot_mini_ticker(symbol, d, data)

    def _process_futures_message(self, data: Dict):
        channel = data.get("channel", "")
        symbol = data.get("symbol", "")
        d = data.get("data", data)

        if "kline" in channel:
            self._handle_futures_kline(symbol, d, data)
        elif "ticker" in channel:
            self._handle_futures_ticker(symbol, d, data)
        elif "depth" in channel:
            self._handle_futures_depth(symbol, d, data)

    def _handle_spot_kline(self, symbol: str, d: Dict, raw: Dict):
        key = symbol.lower().replace("/", "")
        interval = d.get("i", "1m")
        cache_key = f"{key}_{interval}"

        candle = {
            "time": int(d.get("t", time.time() * 1000)),
            "open": float(d.get("o", 0)),
            "high": float(d.get("h", 0)),
            "low": float(d.get("l", 0)),
            "close": float(d.get("c", 0)),
            "volume": float(d.get("v", 0)),
            "closed": d.get("x", False),
        }

        with self._lock:
            if cache_key not in self._candles:
                self._candles[cache_key] = deque(maxlen=self._max_candles)
            candles = self._candles[cache_key]
            if candle["closed"]:
                if candles and candles[-1]["time"] == candle["time"]:
                    candles[-1] = candle
                else:
                    candles.append(candle)
            else:
                if candles and candles[-1]["time"] == candle["time"]:
                    candles[-1] = candle
                else:
                    candles.append(candle)

        self._emit("candle", symbol, interval, candle)

    def _handle_futures_kline(self, symbol: str, d: Dict, raw: Dict):
        self._handle_spot_kline(symbol, {
            "t": d.get("t", d.get("timestamp", int(time.time() * 1000))),
            "o": d.get("o", d.get("open", 0)),
            "h": d.get("h", d.get("high", 0)),
            "l": d.get("l", d.get("low", 0)),
            "c": d.get("c", d.get("close", 0)),
            "v": d.get("v", d.get("vol", 0)),
            "x": d.get("x", True),
            "i": d.get("interval", "1m"),
        }, raw)

    def _handle_spot_ticker(self, symbol: str, d: Dict, raw: Dict):
        ticker = {
            "symbol": symbol,
            "bid": float(d.get("b", 0)),
            "ask": float(d.get("a", 0)),
            "bid_qty": float(d.get("B", 0)),
            "ask_qty": float(d.get("A", 0)),
            "timestamp": int(time.time() * 1000),
        }
        with self._lock:
            self._tickers[symbol] = ticker
        self._emit("ticker", symbol, ticker)

    def _handle_futures_ticker(self, symbol: str, d: Dict, raw: Dict):
        ticker = {
            "symbol": symbol,
            "bid": float(d.get("bid1", 0)),
            "ask": float(d.get("ask1", 0)),
            "bid_qty": float(d.get("bid1Vol", 0)),
            "ask_qty": float(d.get("ask1Vol", 0)),
            "last": float(d.get("lastPrice", 0)),
            "timestamp": int(time.time() * 1000),
        }
        with self._lock:
            self._tickers[symbol] = ticker
        self._emit("ticker", symbol, ticker)

    def _handle_spot_depth(self, symbol: str, d: Dict, raw: Dict):
        bids = [(float(b["p"]), float(b["v"])) for b in d.get("bids", d.get("b", []))]
        asks = [(float(a["p"]), float(a["v"])) for a in d.get("asks", d.get("a", []))]
        ob = {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": int(time.time() * 1000)}
        with self._lock:
            self._orderbooks[symbol] = ob
        self._emit("depth", symbol, ob)

    def _handle_futures_depth(self, symbol: str, d: Dict, raw: Dict):
        bids = [(float(b.get("p", 0)), float(b.get("v", 0))) for b in d.get("bids", [])]
        asks = [(float(a.get("p", 0)), float(a.get("v", 0))) for a in d.get("asks", [])]
        ob = {"symbol": symbol, "bids": bids, "asks": asks, "timestamp": int(time.time() * 1000)}
        with self._lock:
            self._orderbooks[symbol] = ob
        self._emit("depth", symbol, ob)

    def _handle_spot_mini_ticker(self, symbol: str, d: Dict, raw: Dict):
        ticker = {
            "symbol": symbol,
            "last": float(d.get("c", d.get("p", 0))),
            "bid": float(d.get("c", d.get("p", 0))),
            "ask": float(d.get("c", d.get("p", 0))),
            "timestamp": int(time.time() * 1000),
        }
        with self._lock:
            self._tickers[symbol] = ticker
        self._emit("miniticker", symbol, ticker)

    def close(self):
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        print("MEXC WebSocket closed.")
