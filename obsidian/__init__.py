"""Obsidian vault integration for MEXC Quant Trading.

Writes trade logs, daily summaries, and KPI reports to the user's
local Obsidian vault (TTRG convention).
"""

from obsidian.config import ObsidianConfig, get_obsidian_config
from obsidian.vault_detector import detect_vault, vault_exists
from obsidian.trade_logger import TradeLogger, get_trade_logger
from obsidian.daily_summary import DailySummary, get_daily_summary
from obsidian.index_updater import IndexUpdater

__all__ = [
    "ObsidianConfig",
    "get_obsidian_config",
    "detect_vault",
    "vault_exists",
    "TradeLogger",
    "get_trade_logger",
    "DailySummary",
    "get_daily_summary",
    "IndexUpdater",
]
