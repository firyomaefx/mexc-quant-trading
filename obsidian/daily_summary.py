"""Daily summary generator: aggregates CSV trades and writes daily report."""

from __future__ import annotations

import os
import csv
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from collections import defaultdict

from obsidian.config import ObsidianConfig, get_obsidian_config
from obsidian.markdown_writer import write_daily_summary, write_kpi_to_vault_root
from obsidian.trade_logger import get_trade_logger


class DailySummary:
    def __init__(self, cfg: Optional[ObsidianConfig] = None):
        self.cfg = cfg or get_obsidian_config()
        self.logger = get_trade_logger()

    def aggregate(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        trades_raw = self.logger.read_today_trades(target_date)
        target = target_date or date.today().strftime("%Y-%m-%d")

        n = len(trades_raw)
        if n == 0:
            return {
                "date": target,
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "gross_pnl": 0.0,
                "commission": 0.0,
                "net_pnl": 0.0,
                "net_pnl_pct": 0.0,
                "profit_factor": 0.0,
                "pairs": {},
                "best_trade": {},
                "worst_trade": {},
                "capital_start": 18.0,
                "capital_end": 18.0,
            }

        wins, losses, gross, commission, net = 0, 0, 0.0, 0.0, 0.0
        pairs: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "trades": 0, "wins": 0, "pnl": 0.0, "win_rate": 0.0
        })
        best, worst = None, None
        gross_wins, gross_losses = 0.0, 0.0

        for t in trades_raw:
            try:
                pnl = float(t.get("pnl", 0.0))
                comm = float(t.get("commission", 0.0))
                gr = float(t.get("gross_pnl", 0.0))
            except (ValueError, TypeError):
                continue

            if pnl > 0:
                wins += 1
                gross_wins += pnl
            else:
                losses += 1
                gross_losses += abs(pnl)

            gross += gr
            commission += comm
            net += pnl

            sym = t.get("symbol", "UNKNOWN")
            pairs[sym]["trades"] += 1
            pairs[sym]["pnl"] += pnl
            if pnl > 0:
                pairs[sym]["wins"] += 1

            if best is None or pnl > best.get("pnl", -1e9):
                best = {
                    "symbol": sym, "side": t.get("side", ""),
                    "pnl": pnl, "pnl_pct": float(t.get("pnl_pct", 0.0)),
                }
            if worst is None or pnl < worst.get("pnl", 1e9):
                worst = {
                    "symbol": sym, "side": t.get("side", ""),
                    "pnl": pnl, "pnl_pct": float(t.get("pnl_pct", 0.0)),
                }

        for sym, p in pairs.items():
            p["win_rate"] = p["wins"] / p["trades"] if p["trades"] else 0.0

        pf = gross_wins / gross_losses if gross_losses > 0 else 0.0
        cap_start = 18.0
        cap_end = cap_start + net
        return {
            "date": target,
            "trades": n,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / n if n else 0.0,
            "gross_pnl": gross,
            "commission": commission,
            "net_pnl": net,
            "net_pnl_pct": (net / cap_start) * 100.0,
            "profit_factor": pf,
            "pairs": dict(pairs),
            "best_trade": best or {},
            "worst_trade": worst or {},
            "capital_start": cap_start,
            "capital_end": cap_end,
        }

    def write_daily(self, target_date: Optional[str] = None) -> Optional[str]:
        if not self.cfg.enabled or not self.cfg.write_daily_summary:
            return None
        ok, _ = self.cfg.is_valid()
        if not ok:
            return None

        summary = self.aggregate(target_date)
        out_path = os.path.join(self.cfg.summaries_dir(), f"{summary['date']}.md")
        try:
            write_daily_summary(out_path, summary)
        except OSError as e:
            print(f"[DailySummary] Markdown write failed: {e}")
            return None

        if self.cfg.write_kpi:
            kpi_path = os.path.join(self.cfg.kpi_dir(), f"{summary['date']}.md")
            try:
                write_kpi_to_vault_root(kpi_path, summary)
            except OSError as e:
                print(f"[DailySummary] KPI write failed: {e}")

        return out_path


_summary_instance: Optional[DailySummary] = None


def get_daily_summary() -> DailySummary:
    global _summary_instance
    if _summary_instance is None:
        _summary_instance = DailySummary()
    return _summary_instance
