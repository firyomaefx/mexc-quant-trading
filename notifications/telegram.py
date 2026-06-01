"""Telegram notification helpers for trade alerts and daily summaries."""

from __future__ import annotations

import os
import json
import threading
import urllib.request
import urllib.parse
from typing import Optional, Dict, Any
from datetime import datetime


_GLOBAL_LOCK = threading.Lock()
_LAST_SENT_TS: Dict[str, float] = {}
_MIN_INTERVAL_S = 1.0


def _bot_token() -> str:
    return os.environ.get("TELEGRAM_BOT_TOKEN", "")


def _chat_id() -> str:
    return os.environ.get("TELEGRAM_CHAT_ID", "")


def is_configured() -> bool:
    return bool(_bot_token() and _chat_id())


telegram_is_configured = is_configured


def _send_raw(text: str) -> bool:
    if not is_configured():
        return False
    try:
        url = f"https://api.telegram.org/bot{_bot_token()}/sendMessage"
        payload = urllib.parse.urlencode({
            "chat_id": _chat_id(),
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode("utf-8", errors="ignore")
        try:
            obj = json.loads(data)
            return bool(obj.get("ok"))
        except (ValueError, TypeError):
            return False
    except Exception:
        return False


def _send_async(text: str) -> None:
    def _runner():
        with _GLOBAL_LOCK:
            now_ts = datetime.now().timestamp()
            last = _LAST_SENT_TS.get("last", 0.0)
            if now_ts - last < _MIN_INTERVAL_S:
                return
            _LAST_SENT_TS["last"] = now_ts
        _send_raw(text)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()


def send_trade_alert(trade: Dict[str, Any]) -> bool:
    if not is_configured():
        return False
    sym = trade.get("symbol", "?")
    side = trade.get("side", trade.get("direction", "?"))
    pnl = float(trade.get("pnl", 0.0))
    pnl_pct = float(trade.get("pnl_pct", 0.0))
    entry = float(trade.get("entry_price", 0.0))
    exit_p = float(trade.get("exit_price", 0.0))
    reason = trade.get("exit_reason", trade.get("reason", "?"))
    emoji = "+" if pnl >= 0 else ""
    arrow = "TP" if pnl >= 0 else "SL"
    text = (
        f"<b>{arrow} {sym} {side}</b>\n"
        f"Entry: <code>{entry:.4f}</code>  Exit: <code>{exit_p:.4f}</code>\n"
        f"P&amp;L: <b>${emoji}{pnl:+.4f}</b> ({pnl_pct:+.2f}%)\n"
        f"Reason: <i>{reason}</i>"
    )
    _send_async(text)
    return True


def send_daily_summary(summary: Dict[str, Any]) -> bool:
    if not is_configured():
        return False
    date = summary.get("date", datetime.now().strftime("%Y-%m-%d"))
    n = int(summary.get("trades", 0))
    wins = int(summary.get("wins", 0))
    losses = int(summary.get("losses", 0))
    wr = float(summary.get("win_rate", 0.0)) * 100
    net = float(summary.get("net_pnl", 0.0))
    cap = float(summary.get("capital_end", 0.0))
    pf = float(summary.get("profit_factor", 0.0))
    emoji = "UP" if net >= 0 else "DOWN"
    text = (
        f"<b>DAILY {date} {emoji}</b>\n"
        f"Trades: {n}  W/L: {wins}/{losses}  WR: {wr:.0f}%\n"
        f"Net: <b>${net:+.4f}</b>  Capital: <b>${cap:.2f}</b>\n"
        f"PF: {pf:.2f}"
    )
    _send_async(text)
    return True


def send_circuit_breaker_alert(reason: str, is_halted: bool, daily_loss_pct: float = 0.0) -> bool:
    if not is_configured():
        return False
    state = "HALTED" if is_halted else "WARNING"
    text = (
        f"<b>CIRCUIT BREAKER {state}</b>\n"
        f"Reason: <i>{reason}</i>\n"
        f"Daily Loss: {daily_loss_pct:.2f}%"
    )
    _send_async(text)
    return True


telegram_is_configured = is_configured
telegram_send_trade = send_trade_alert
telegram_send_summary = send_daily_summary
telegram_send_breaker = send_circuit_breaker_alert
