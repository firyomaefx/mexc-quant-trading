"""24-hour continuous paper trading test.

Runs `LivePaperEngine` for 24 hours, logging every trade to:
  - trade_journal.csv
  - Obsidian vault (auto-sync)
  - Telegram alerts (if configured)
  - dashboard/state.json

Usage:
  python run_24h_test.py [--hours 24] [--interval 30] [--no-obsidian] [--no-telegram]
"""
import os
import sys
import time
import signal
import argparse
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

from config.crypto_config import CRYPTO_CONFIG
from config.pairs import list_enabled_symbols
from live.mexc_hybrid import MEXCHybridConnector
from live.live_paper import LivePaperEngine


def main():
    parser = argparse.ArgumentParser(description="24-hour continuous paper trading test")
    parser.add_argument("--hours", type=float, default=24.0, help="Duration in hours (default 24)")
    parser.add_argument("--interval", type=int, default=30, help="Poll interval in seconds (default 30)")
    parser.add_argument("--no-obsidian", action="store_true", help="Disable Obsidian sync")
    parser.add_argument("--no-telegram", action="store_true", help="Disable Telegram alerts")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  24-HOUR CONTINUOUS PAPER TEST")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Duration: {args.hours} hours")
    print(f"  Interval: {args.interval}s")
    print(f"  Obsidian: {'OFF' if args.no_obsidian else 'ON'}")
    print(f"  Telegram: {'OFF' if args.no_telegram else 'ON'}")
    print("=" * 70)

    cfg = CRYPTO_CONFIG
    if args.no_obsidian:
        cfg.obsidian.enabled = False
    if args.no_telegram:
        cfg.notifications.enabled = False

    api_key = os.getenv("MEXC_API_KEY", "")
    api_secret = os.getenv("MEXC_API_SECRET", "")
    proxy = os.getenv("MEXC_PROXY", "")

    mexc = MEXCHybridConnector(api_key, api_secret, proxy=proxy, use_coingecko=True)
    mexc.connect()

    engine = LivePaperEngine(cfg, mexc)
    engine.load_initial_data()
    print(f"\n  Pairs: {list_enabled_symbols(cfg)}")
    print(f"  Capital: ${cfg.scalping.initial_capital:.2f}")
    print(f"  ML enabled: {cfg.ml.enabled}")
    print(f"  LLM sentiment: {cfg.llm_sentiment.enabled}")
    print(f"  Obsidian vault: {cfg.obsidian.vault_path}")
    print()

    start = datetime.now()
    end = start + timedelta(hours=args.hours)
    print(f"  Will run until: {end.isoformat()}")
    print(f"  Press Ctrl+C to stop early\n")

    def _signal_handler(sig, frame):
        print("\n  Stopping engine...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    try:
        engine.run_loop(interval_seconds=args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        elapsed = datetime.now() - start
        print(f"\n  Total elapsed: {elapsed}")
        print(f"  Final equity: ${engine._equity:.2f}")
        print(f"  Trades: {len(engine._trade_history)}")
        if engine._trade_history:
            wins = [t for t in engine._trade_history if t.get("pnl", 0) > 0]
            total_pnl = sum(t.get("pnl", 0) for t in engine._trade_history)
            print(f"  Win rate: {len(wins)/len(engine._trade_history)*100:.1f}%")
            print(f"  Net P&L: ${total_pnl:+.4f}")


if __name__ == "__main__":
    main()
