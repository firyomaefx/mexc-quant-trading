import sys
import os
_quant_v2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_parent = os.path.dirname(_quant_v2)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

import json
import time
from typing import Optional, Dict
from datetime import datetime

from quant_v2.config.crypto_config import SentimentConfig


class LLMSentimentAnalyzer:
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self._cache: Dict[str, tuple] = {}
        self._last_fetch = 0.0
        self._cached_value = 0.0
        self._openai_client = None
        self._use_ollama = False

        openai_key = os.getenv("OPENAI_API_KEY", "")
        ollama_host = os.getenv("OLLAMA_HOST", "")
        if openai_key:
            try:
                from openai import OpenAI
                self._openai_client = OpenAI(api_key=openai_key)
                print("LLM Sentiment: OpenAI GPT-4o-mini connected")
            except ImportError:
                print("LLM Sentiment: openai package not installed, using keyword fallback")
        elif ollama_host:
            self._use_ollama = True
            print(f"LLM Sentiment: Using Ollama at {ollama_host}")

    def get_sentiment(self, symbol: str = "BTC") -> float:
        now = time.time()
        if now - self._last_fetch < self.config.cache_seconds:
            return self._cached_value

        news_headlines = self._fetch_news_headlines(symbol)
        if not news_headlines:
            from quant_v2.signals.sentiment import SentimentAnalyzer
            fallback = SentimentAnalyzer(self.config)
            return fallback.get_sentiment(symbol)

        llm_score = self._analyze_with_llm(news_headlines, symbol)
        if llm_score is not None:
            self._cached_value = llm_score
            self._last_fetch = now
            return llm_score

        keyword_score = self._analyze_with_keywords(news_headlines)
        self._cached_value = keyword_score
        self._last_fetch = now
        return keyword_score

    def _fetch_news_headlines(self, symbol: str) -> list:
        cache_key = f"news_{symbol}"
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self.config.cache_seconds:
            return cached[1]

        headlines = []
        try:
            import urllib.request
            url = f"{self.config.cryptopanic_url}?public=true&currencies={symbol.split('/')[0]}&kind=news"
            if self.config.cryptopanic_api_key:
                url += f"&auth_token={self.config.cryptopanic_api_key}"
            req = urllib.request.Request(url, headers={"User-Agent": "QuantV2/2.0"})
            resp = urllib.request.urlopen(req, timeout=10)
            data = json.loads(resp.read().decode())
            results = data.get("results", [])[:15]
            for post in results:
                title = post.get("title", "")
                if title:
                    headlines.append({
                        "title": title,
                        "published": post.get("published_at", ""),
                        "url": post.get("url", ""),
                    })
        except Exception:
            pass

        self._cache[cache_key] = (time.time(), headlines)
        return headlines

    def _analyze_with_llm(self, headlines: list, symbol: str) -> Optional[float]:
        if not headlines:
            return None

        headline_text = "\n".join([f"- {h['title']}" for h in headlines[:10]])
        prompt = (
            f"Analyze these {symbol} crypto news headlines for market sentiment. "
            f"Consider sarcasm, negation, and context.\n\n"
            f"Headlines:\n{headline_text}\n\n"
            f"Respond ONLY with a JSON object:\n"
            f'{{"sentiment": <float -1.0 to +1.0>, "confidence": <float 0-1>, '
            f'"reasoning": "<10 word summary>"}}'
        )

        if self._openai_client:
            return self._call_openai(prompt)
        elif self._use_ollama:
            return self._call_ollama(prompt)
        return None

    def _call_openai(self, prompt: str) -> Optional[float]:
        try:
            response = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a crypto sentiment analyzer. Respond only with the JSON object requested."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()
            return self._parse_llm_response(content)
        except Exception as e:
            print(f"OpenAI sentiment error: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[float]:
        import urllib.request
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        try:
            data = json.dumps({
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 100},
            }).encode()
            req = urllib.request.Request(
                f"{ollama_host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=15)
            result = json.loads(resp.read().decode())
            content = result.get("response", "")
            return self._parse_llm_response(content)
        except Exception as e:
            print(f"Ollama sentiment error: {e}")
            return None

    def _parse_llm_response(self, content: str) -> Optional[float]:
        try:
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            parsed = json.loads(content)
            sentiment = float(parsed.get("sentiment", 0))
            sentiment = max(-1.0, min(1.0, sentiment))
            return sentiment
        except (json.JSONDecodeError, ValueError, TypeError):
            try:
                import re
                numbers = re.findall(r"[-+]?\d+\.\d+", content)
                if numbers:
                    return max(-1.0, min(1.0, float(numbers[0])))
            except Exception:
                pass
            return None

    def _analyze_with_keywords(self, headlines: list) -> float:
        if not headlines:
            return 0.0

        bullish_words = {
            "surge", "rally", "bullish", "breakout", "pump", "moon",
            "buy", "long", "gain", "rise", "upgrade", "adoption",
            "etf", "institutional", "partnership", "launch", "milestone",
        }
        bearish_words = {
            "crash", "dump", "bearish", "collapse", "sell-off", "decline",
            "drop", "warning", "ban", "hack", "regulation", "fraud",
            "rug", "scam", "lawsuit", "restrict", "risk",
        }

        total = 0
        positive = 0
        negative = 0

        for h in headlines:
            title = h.get("title", "").lower()
            words = set(title.split())
            bull = len(words & bullish_words)
            bear = len(words & bearish_words)
            positive += bull
            negative += bear
            total += max(1, bull + bear)

        if total == 0:
            return 0.0

        score = (positive - negative) / total
        return max(-1.0, min(1.0, score))

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