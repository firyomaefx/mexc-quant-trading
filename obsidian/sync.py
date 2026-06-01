"""CLI: python -m obsidian.sync [--trades-only | --summary-only] [--date YYYY-MM-DD]"""

from __future__ import annotations

import argparse
import sys
import os

from obsidian.config import get_obsidian_config
from obsidian.vault_detector import detect_vault, vault_exists
from obsidian.trade_logger import get_trade_logger
from obsidian.daily_summary import get_daily_summary
from obsidian.index_updater import IndexUpdater


def main():
    parser = argparse.ArgumentParser(description="Sync MEXC quant data to Obsidian vault")
    parser.add_argument("--vault", type=str, default=None, help="Vault path (overrides config)")
    parser.add_argument("--date", type=str, default=None, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--trades-only", action="store_true", help="Only sync trade notes")
    parser.add_argument("--summary-only", action="store_true", help="Only sync daily summary")
    parser.add_argument("--detect", action="store_true", help="Detect vault and exit")
    parser.add_argument("--verify", action="store_true", help="Verify config and exit")
    args = parser.parse_args()

    cfg = get_obsidian_config()

    if args.vault:
        cfg.vault_path = args.vault

    if not cfg.enabled:
        print("[obsidian] integration disabled in config")
        return 0

    if args.detect:
        detected = detect_vault(cfg.vault_path)
        if detected:
            print(f"[obsidian] detected vault: {detected}")
        else:
            print("[obsidian] no vault detected")
        return 0

    ok, msg = cfg.is_valid()
    if not ok:
        print(f"[obsidian] config invalid: {msg}")
        print(f"[obsidian] vault_path = {cfg.vault_path}")
        return 1

    if args.verify:
        print(f"[obsidian] OK")
        print(f"  vault: {cfg.vault_path}")
        print(f"  project: {cfg.project_folder}")
        print(f"  trades_dir: {cfg.trades_dir()}")
        print(f"  summaries_dir: {cfg.summaries_dir()}")
        print(f"  kpi_dir: {cfg.kpi_dir()}")
        return 0

    target = args.date
    summary = get_daily_summary()
    updater = IndexUpdater()
    logger = get_trade_logger()

    if not args.summary_only:
        trades = logger.read_today_trades(target)
        if trades:
            written = logger.log_trade({})
            for t in trades:
                trade_dict = {k: t.get(k, "") for k in [
                    "symbol", "side", "entry_price", "exit_price", "qty",
                    "notional", "gross_pnl", "commission", "pnl", "pnl_pct",
                    "duration_bars", "exit_reason", "z_score", "ema_slope",
                    "ml_confidence",
                ]}
                for num_key in ("entry_price", "exit_price", "qty", "notional",
                                "gross_pnl", "commission", "pnl", "pnl_pct",
                                "z_score", "ema_slope", "ml_confidence"):
                    try:
                        trade_dict[num_key] = float(trade_dict.get(num_key, 0.0))
                    except (ValueError, TypeError):
                        trade_dict[num_key] = 0.0
                try:
                    trade_dict["duration_bars"] = int(trade_dict.get("duration_bars", 0))
                except (ValueError, TypeError):
                    trade_dict["duration_bars"] = 0
                trade_dict["entry_time"] = t.get("timestamp", "")
                trade_dict["exit_time"] = t.get("timestamp", "")
                trade_dict["date"] = t.get("date", target or "")
                logger.log_trade(trade_dict)
            updater.update_trade_index(target)
            print(f"[obsidian] wrote {len(trades)} trade notes")
        else:
            print(f"[obsidian] no trades to sync for {target or 'today'}")

    if not args.trades_only:
        path = summary.write_daily(target)
        if path:
            print(f"[obsidian] wrote daily summary: {path}")
            agg = summary.aggregate(target)
            updater.update_summary_index(target)
            updater.update_trade_journal(agg)
        else:
            print(f"[obsidian] no summary written (check config or no trades)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
