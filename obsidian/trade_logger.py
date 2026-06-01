"""Trade logger: appends closed trades to per-day markdown files in the vault."""

from __future__ import annotations

import os
import csv
import json
import threading
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from obsidian.config import ObsidianConfig, get_obsidian_config
from obsidian.markdown_writer import write_trade_note
from obsidian.vault_detector import vault_exists


DEFAULT_CSV = "trade_journal.csv"


class TradeLogger:
    def __init__(self, cfg: Optional[ObsidianConfig] = None, csv_path: Optional[str] = None):
        self.cfg = cfg or get_obsidian_config()
        self.csv_path = csv_path or self._default_csv_path()
        self._lock = threading.Lock()
        self._cached_trade_count = 0
        self._ensure_csv()

    def _default_csv_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, DEFAULT_CSV)

    def _ensure_csv(self) -> None:
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "date", "symbol", "side",
                    "entry_price", "exit_price", "qty", "notional",
                    "gross_pnl", "commission", "pnl", "pnl_pct",
                    "duration_bars", "exit_reason",
                    "z_score", "ema_slope", "ml_confidence",
                ])

    def _append_csv(self, trade: Dict[str, Any]) -> None:
        try:
            with self._lock:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().isoformat(timespec="seconds"),
                        trade.get("date", datetime.now().strftime("%Y-%m-%d")),
                        trade.get("symbol", ""),
                        trade.get("side", ""),
                        f"{float(trade.get('entry_price', 0.0)):.8f}",
                        f"{float(trade.get('exit_price', 0.0)):.8f}",
                        f"{float(trade.get('qty', 0.0)):.8f}",
                        f"{float(trade.get('notional', 0.0)):.4f}",
                        f"{float(trade.get('gross_pnl', 0.0)):.6f}",
                        f"{float(trade.get('commission', 0.0)):.6f}",
                        f"{float(trade.get('pnl', 0.0)):.6f}",
                        f"{float(trade.get('pnl_pct', 0.0)):.6f}",
                        int(trade.get("duration_bars", 0)),
                        trade.get("exit_reason", ""),
                        f"{float(trade.get('z_score', 0.0)):.4f}",
                        f"{float(trade.get('ema_slope', 0.0)):.6f}",
                        f"{float(trade.get('ml_confidence', 0.0)):.4f}",
                    ])
        except OSError as e:
            print(f"[TradeLogger] CSV write failed: {e}")

    def log_trade(self, trade: Dict[str, Any]) -> Optional[str]:
        if not self.cfg.enabled or not self.cfg.write_trades:
            return None
        ok, _ = self.cfg.is_valid()
        if not ok:
            return None

        self._append_csv(trade)

        today = trade.get("date", datetime.now().strftime("%Y-%m-%d"))
        out_path = os.path.join(self.cfg.trades_dir(), f"{today}.md")
        try:
            return write_trade_note(out_path, trade)
        except OSError as e:
            print(f"[TradeLogger] Markdown write failed: {e}")
            return None

    def read_today_trades(self, target_date: Optional[str] = None) -> List[Dict[str, Any]]:
        target = target_date or date.today().strftime("%Y-%m-%d")
        trades = []
        try:
            with open(self.csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("date") == target:
                        trades.append(row)
        except (OSError, KeyError):
            return []
        return trades

    def has_written_today(self, target_date: Optional[str] = None) -> bool:
        target = target_date or date.today().strftime("%Y-%m-%d")
        out_path = os.path.join(self.cfg.trades_dir(), f"{target}.md")
        return os.path.isfile(out_path)


_logger_instance: Optional[TradeLogger] = None


def get_trade_logger() -> TradeLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = TradeLogger()
    return _logger_instance
