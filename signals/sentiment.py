import os
import sys
import sys

import json
import time
import urllib.request
from typing import Optional, Dict
from datetime import datetime, timedelta

from config.crypto_config import SentimentConfig


class SentimentAnalyzer:
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self._cache: Dict[str, tuple] = {}
        self._last_fetch = 0.0
        self._cached_value = 0.0

    def get_sentiment(self, symbol: str = "BTC") -> float:
        now = time.time()
        if now - self._last_fetch < self.config.cache_seconds:
            return self._cached_value

        score = 0.0
        weight_total = 0.0

        if self.config.fear_greed_weight > 0:
            fg = self._fetch_fear_greed()
            if fg is not None:
                score += (fg / 100.0 - 0.5) * 2.0 * self.config.fear_greed_weight
                weight_total += self.config.fear_greed_weight

        if self.config.news_weight > 0 and self.config.cryptopanic_api_key:
            ns = self._fetch_crypto_panic()
            if ns is not None:
                score += ns * self.config.news_weight
                weight_total += self.config.news_weight

        if self.config.price_action_weight > 0:
            pa = self._fetch_price_action_sentiment(symbol)
            if pa is not None:
                score += pa * self.config.price_action_weight
                weight_total += self.config.price_action_weight

        if weight_total > 0:
            score /= weight_total

        score = max(-1.0, min(1.0, score))
        self._last_fetch = now
        self._cached_value = score
        return score

    def _fetch_fear_greed(self) -> Optional[float]:
        cache_key = "fear_greed"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.config.cache_seconds:
            return cached[1]

        try:
            req = urllib.request.Request(
                self.config.fear_greed_url,
                headers={"User-Agent": "QuantV2/1.0"}
            )
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            value = float(data.get("data", [{}])[0].get("value", 50))
            self._cache[cache_key] = (time.time(), value)
            return value
        except Exception as e:
            print(f"Fear & Greed fetch failed: {e}")
            return None

    def _fetch_crypto_panic(self) -> Optional[float]:
        cache_key = "crypto_panic"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.config.cache_seconds:
            return cached[1]

        try:
            url = f"{self.config.cryptopanic_url}?auth_token={self.config.cryptopanic_api_key}&currencies=BTC,ETH,XRP,ADA,SOL&filter=important&kind=news"
            req = urllib.request.Request(url, headers={"User-Agent": "QuantV2/1.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())

            results = data.get("results", [])
            if not results:
                return 0.0

            total_votes = 0
            positive_votes = 0
            negative_votes = 0

            for post in results[:20]:
                votes = post.get("votes", {})
                pos = votes.get("positive", 0)
                neg = votes.get("negative", 0)
                important = votes.get("important", 0)
                positive_votes += pos + important
                negative_votes += neg
                total_votes += pos + neg + important

                title = (post.get("title", "") + " " + post.get("body", "")).lower()
                bullish_words = ["surge", "rally", "bullish", "breakout", "pump", "moon", "buy", "long"]
                bearish_words = ["crash", "dump", "bearish", "collapse", "sell-off", "decline", "drop", "warning"]

                bull_count = sum(1 for w in bullish_words if w in title)
                bear_count = sum(1 for w in bearish_words if w in title)
                positive_votes += bull_count * 3
                negative_votes += bear_count * 3
                total_votes += (bull_count + bear_count) * 3

            if total_votes == 0:
                return 0.0

            score = (positive_votes - negative_votes) / total_votes
            self._cache[cache_key] = (time.time(), score)
            return score

        except Exception as e:
            print(f"CryptoPanic fetch failed: {e}")
            return None

    def _fetch_price_action_sentiment(self, symbol: str) -> Optional[float]:
        cache_key = f"price_action_{symbol}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < 60:
            return cached[1]

        try:
            import ccxt
            exchange = ccxt.mexc({"enableRateLimit": True})
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="15m", limit=20)
            if not ohlcv:
                return 0.0

            closes = [c[4] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]

            sma5 = sum(closes[-5:]) / 5
            sma20 = sum(closes[-20:]) / min(20, len(closes))
            sma_diff = (sma5 / sma20 - 1.0) * 5.0 if sma20 > 0 else 0.0

            avg_vol = sum(volumes[-10:]) / min(10, len(volumes))
            recent_vol = sum(volumes[-3:]) / 3
            vol_ratio = (recent_vol / avg_vol - 1.0) * 2.0 if avg_vol > 0 else 0.0

            body_sizes = []
            for i in range(len(ohlcv) - 5, len(ohlcv)):
                body = abs(closes[i] - ohlcv[i][1])
                if ohlcv[i][2] != ohlcv[i][3]:
                    body /= (ohlcv[i][2] - ohlcv[i][3])
                body_sizes.append(min(body, 2.0))
            avg_body = sum(body_sizes) / len(body_sizes) if body_sizes else 0.5

            score = sma_diff * 0.4 + vol_ratio * 0.2 + (1.0 - avg_body) * 0.2
            score = max(-1.0, min(1.0, score))

            self._cache[cache_key] = (time.time(), score)
            return score

        except Exception as e:
            print(f"Price action sentiment failed for {symbol}: {e}")
            return None

    def get_sentiment_label(self, score: float) -> str:
        if score > 0.3:
            return "bullish"
        elif score < -0.3:
            return "bearish"
        elif score > 0.05:
            return "slightly_bullish"
        elif score < -0.05:
            return "slightly_bearish"
        return "neutral"

    def get_signal_bias(self, score: float, base_signal: int) -> float:
        if base_signal == 1:
            return 1.0 + score * 0.3
        elif base_signal == -1:
            return 1.0 - score * 0.3
        return 1.0