"""Notification module exports."""

from notifications.telegram import (
    is_configured as telegram_is_configured,
    send_trade_alert as telegram_send_trade,
    send_daily_summary as telegram_send_summary,
    send_circuit_breaker_alert as telegram_send_breaker,
)

__all__ = [
    "telegram_is_configured",
    "telegram_send_trade",
    "telegram_send_summary",
    "telegram_send_breaker",
]
