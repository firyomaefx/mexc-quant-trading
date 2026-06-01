"""Index updater: appends [[wikilinks]] to project Index.md and trade-day list."""

from __future__ import annotations

import os
import re
from datetime import datetime, date
from typing import Optional

from obsidian.config import ObsidianConfig, get_obsidian_config


class IndexUpdater:
    def __init__(self, cfg: Optional[ObsidianConfig] = None):
        self.cfg = cfg or get_obsidian_config()

    def _insert_into_index(self, index_path: str, new_line: str, marker: str) -> bool:
        if not os.path.isfile(index_path):
            return False
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return False

        if new_line.strip() in content:
            return False

        pattern = re.compile(
            rf"({re.escape(marker)}\n)(.*?)(?=\n##|\Z)",
            re.DOTALL,
        )
        m = pattern.search(content)
        if not m:
            return False

        block = m.group(2)
        lines = block.rstrip("\n").split("\n")
        if not lines or lines[0].strip() == "":
            new_block = new_line + "\n"
        else:
            new_block = "\n".join(lines) + "\n" + new_line + "\n"

        new_content = content[:m.start(2)] + new_block + content[m.end(2):]
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        except OSError:
            return False

    def update_trade_index(self, target_date: Optional[str] = None) -> bool:
        if not self.cfg.enabled or not self.cfg.update_project_index:
            return False
        ok, _ = self.cfg.is_valid()
        if not ok:
            return False
        target = target_date or date.today().strftime("%Y-%m-%d")
        index_path = os.path.join(self.cfg.project_dir(), "trades", "index.md")
        new_line = f"- [[{target}|{target}]]"
        return self._insert_into_index(index_path, new_line, "## Archive")

    def update_summary_index(self, target_date: Optional[str] = None) -> bool:
        if not self.cfg.enabled or not self.cfg.update_project_index:
            return False
        ok, _ = self.cfg.is_valid()
        if not ok:
            return False
        target = target_date or date.today().strftime("%Y-%m-%d")
        index_path = os.path.join(self.cfg.project_dir(), "daily-summaries", "index.md")
        new_line = f"- [[{target}|{target}]]"
        return self._insert_into_index(index_path, new_line, "## Archive")

    def update_trade_journal(self, summary: dict) -> bool:
        if not self.cfg.enabled or not self.cfg.update_project_index:
            return False
        ok, _ = self.cfg.is_valid()
        if not ok:
            return False
        index_path = os.path.join(self.cfg.project_dir(), "trade-journal.md")
        if not os.path.isfile(index_path):
            return False

        target = summary.get("date", date.today().strftime("%Y-%m-%d"))
        new_row = (
            f"| {target} | {summary.get('trades', 0)} | "
            f"{summary.get('wins', 0)} | {summary.get('losses', 0)} | "
            f"${summary.get('net_pnl', 0.0):+.4f} | "
            f"{summary.get('win_rate', 0.0):.0%} |"
        )

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError:
            return False

        if new_row in content:
            return False

        pattern = re.compile(
            r"(\| Date \| Trades \| Wins \| Losses \| Net P&L \| Win Rate \|\n\|[-\s|]+\n)(.*?)(?=\n##|\Z)",
            re.DOTALL,
        )
        m = pattern.search(content)
        if not m:
            return False

        block = m.group(2).rstrip("\n")
        if not block.strip() or "—" in block or "TBD" in block:
            new_block = new_row + "\n"
        else:
            new_block = block + "\n" + new_row + "\n"

        new_content = content[:m.start(2)] + new_block + content[m.end(2):]
        try:
            with open(index_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            return True
        except OSError:
            return False
