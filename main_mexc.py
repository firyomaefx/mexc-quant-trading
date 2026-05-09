#!/usr/bin/env python
"""
Quant V2 - AI-Powered Multi-Pair MEXC Scalping Bot

Commands:
  backtest   Run walk-forward backtest on historical data
  paper      Run paper trading simulation
  live       Start live trading
  dashboard  Launch trading dashboard
  optimize   Optimize parameters via grid search
  status     Show current status and risk metrics
  pairs      List configured trading pairs
  sentiment  Check current market sentiment
"""

import sys
import argparse
import json
from datetime import datetime
from typing import Dict, List

_script_dir = os.path.dirname(os.path.abspath(__file__))

from dotenv import load_dotenv
load_dotenv(os.path.join(_parent, ".env"))

from config.crypto_config import CryptoConfig, CRYPTO_CONFIG
from config.pairs import get_pair_config, list_enabled_symbols, list_all_symbols
from live.mexc_adapter import MEXCConnector
from live.mexc_hybrid import MEXCHybridConnector
from live.mexc_futures import MEXCFuturesConnector
from live.paper_trader import PaperTrader
from live.scalper import Scalper
from backtest.walk_forward import WalkForwardBacktester
from backtest.multi_pair_bt import MultiPairBacktester
from signals.sentiment import SentimentAnalyzer


def get_config() -> CryptoConfig:
    return CRYPTO_CONFIG


def get_api_credentials() -> tuple:
    key = os.getenv("MEXC_API_KEY", "")
    secret = os.getenv("MEXC_API_SECRET", "")
    return key, secret


def cmd_backtest(args):
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST")
    print("=" * 70)

    cfg = get_config()
    symbols = args.symbols.split(",") if args.symbols else list_enabled_symbols(cfg)
    print(f"Pairs: {symbols}")
    print(f"Capital: ${cfg.backtest.initial_capital:.0f}")
    print(f"Train: {cfg.backtest.train_months}mo | Test: {cfg.backtest.test_months}mo")
    print(f"Monte Carlo runs: {cfg.backtest.monte_carlo_runs}")

    api_key, api_secret = get_api_credentials()
    proxy = os.getenv("MEXC_PROXY", "")
    mexc = MEXCHybridConnector(api_key, api_secret, proxy=proxy, use_coingecko=True)
    connected = mexc.connect()
    print(f"Data source: {'MEXC Direct' if mexc._mexc_connected else 'CoinGecko (MEXC blocked)'}")

    data = {}
    for sym in symbols:
        print(f"\nLoading {sym}...")
        if connected:
            try:
                df = mexc.fetch_rates(sym, "1m", 1000)
                data[sym] = df
                print(f"  Fetched {len(df)} bars")
            except Exception as e:
                print(f"  Fetch failed: {e}")

    if not data:
        print("\nNo live data available. Generating synthetic data...")
        from data.synthetic import SyntheticDataGenerator
        gen = SyntheticDataGenerator()
        for sym in symbols:
            df = gen.generate_regime_data(n_bars=1000, freq="1min")
            data[sym] = df
            print(f"  {sym}: {len(df)} synthetic bars")

    mexc.disconnect()

    bt = WalkForwardBacktester(cfg)
    results = bt.run_all_pairs(data)
    bt.print_summary()

    mp_bt = MultiPairBacktester(cfg)
    mp_bt._pair_results = results
    portfolio = mp_bt.run(data)
    mp_bt.print_summary(portfolio)

    with open("backtest_results.json", "w") as f:
        json.dump({"walk_forward": results, "portfolio": portfolio}, f, default=str, indent=2)
    print("\nResults saved to backtest_results.json")


def cmd_paper(args):
    cfg = get_config()
    symbols = args.symbols.split(",") if args.symbols else list_enabled_symbols(cfg)
    max_bars = args.bars or 500

    api_key, api_secret = get_api_credentials()
    proxy = os.getenv("MEXC_PROXY", "")
    mexc = MEXCHybridConnector(api_key, api_secret, proxy=proxy, use_coingecko=True)
    mexc.connect()

    trader = PaperTrader(cfg, mexc)
    report = trader.run_simulation(symbols, max_bars)
    trader.print_report(report)

    if args.save:
        with open(f"paper_report_{datetime.now():%Y%m%d_%H%M%S}.json", "w") as f:
            json.dump(report, f, default=str, indent=2)


def cmd_live(args):
    print("\n" + "!" * 70)
    print("  WARNING: LIVE TRADING WITH REAL MONEY")
    print("  This will execute actual trades on your MEXC account.")
    print("  Run 'paper' mode first to validate your strategy.")
    print("!" * 70)

    api_key, api_secret = get_api_credentials()
    if not api_key or not api_secret:
        print("\nERROR: MEXC_API_KEY and MEXC_API_SECRET must be set in .env file")
        print("  Create .env file based on .env.example:")
        print("  MEXC_API_KEY=your_api_key_here")
        print("  MEXC_API_SECRET=your_api_secret_here")
        return

    if not args.yes:
        confirm = input("\nType 'YES' to confirm live trading: ")
        if confirm.upper() != "YES":
            print("Aborted.")
            return

    cfg = get_config()
    if args.futures:
        cfg.futures.enabled = True
        if args.leverage:
            cfg.futures.max_leverage = args.leverage

    proxy = os.getenv("MEXC_PROXY", args.proxy if hasattr(args, 'proxy') else "")
    scalper = Scalper(cfg, api_key, api_secret, args.testnet, use_hybrid=True, proxy=proxy)
    symbols = args.symbols.split(",") if args.symbols else list_enabled_symbols(cfg)
    scalper.run()


def cmd_sentiment(args):
    cfg = get_config()
    analyzer = SentimentAnalyzer(cfg.sentiment)

    print("\n" + "=" * 50)
    print("  MARKET SENTIMENT ANALYSIS")
    print("=" * 50)

    for sym in (args.symbols.split(",") if args.symbols else ["BTC/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT"]):
        score = analyzer.get_sentiment(sym)
        label = analyzer.get_sentiment_label(score)
        bar = "#" * int(abs(score) * 20)
        direction = "+" if score >= 0 else "-"
        print(f"  {sym:>12}: {score:+.2f} [{direction}{bar}] ({label})")

    print("=" * 50 + "\n")


def cmd_pairs(args):
    cfg = get_config()
    print("\n" + "=" * 60)
    print("  CONFIGURED TRADING PAIRS")
    print("=" * 60)

    all_pairs = cfg.pairs or {}
    for sym, pc in all_pairs.items():
        status = "ENABLED" if pc.enabled else "DISABLED"
        print(f"\n  {sym} ({status}):")
        print(f"    Min Qty: {pc.min_qty}  |  Step: {pc.qty_step}")
        print(f"    Z-Score Entry: {pc.zscore_entry_long} / {pc.zscore_entry_short}")
        print(f"    Z-Score Stop:  {pc.zscore_stop_long} / {pc.zscore_stop_short}")
        print(f"    Time Stop: {pc.time_stop_bars} bars  |  Spread: {pc.typical_spread_pct}%")

    print(f"\n  Scalping Config:")
    print(f"    TF: {cfg.scalping.primary_tf}m  |  Max Positions: {cfg.scalping.max_concurrent_positions}")
    print(f"    Risk/Trade: {cfg.scalping.account_risk_pct*100:.1f}%  |  Max Daily Loss: {cfg.scalping.max_daily_loss_pct*100:.0f}%")
    print(f"    Max DD: {cfg.scalping.max_drawdown_pct*100:.0f}%  |  Capital: ${cfg.scalping.initial_capital:.0f}")
    print(f"    Futures: {'ON' if cfg.futures.enabled else 'OFF'}  |  Leverage: {cfg.futures.max_leverage}x")
    print("=" * 60 + "\n")


def cmd_dashboard(args):
    import subprocess
    dashboard_script = os.path.join(os.path.dirname(__file__), "dashboard", "run.py")
    cmd = [sys.executable, dashboard_script, "--mode", args.mode, "--port", str(args.port)]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Quant V2 - MEXC AI-Powered Scalping Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    backtest_parser = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    backtest_parser.add_argument("--symbols", "-s", type=str, default="",
                                 help="Comma-separated symbols (default: all enabled)")
    backtest_parser.add_argument("--capital", type=float, default=0,
                                 help="Initial capital (default: from config)")
    backtest_parser.add_argument("--train", type=int, default=0,
                                 help="Train months (default: from config)")
    backtest_parser.add_argument("--test", type=int, default=0,
                                 help="Test months (default: from config)")

    paper_parser = subparsers.add_parser("paper", help="Run paper trading simulation")
    paper_parser.add_argument("--symbols", "-s", type=str, default="",
                              help="Comma-separated symbols")
    paper_parser.add_argument("--bars", "-b", type=int, default=500,
                              help="Number of bars to simulate")
    paper_parser.add_argument("--save", action="store_true", help="Save report to JSON")

    live_parser = subparsers.add_parser("live", help="Start live trading")
    live_parser.add_argument("--symbols", "-s", type=str, default="",
                             help="Comma-separated symbols")
    live_parser.add_argument("--futures", action="store_true", help="Use futures")
    live_parser.add_argument("--leverage", "-l", type=int, default=0,
                             help="Leverage (futures only)")
    live_parser.add_argument("--testnet", action="store_true", help="Use MEXC testnet")
    live_parser.add_argument("--proxy", type=str, default="", help="Proxy URL (e.g. socks5://127.0.0.1:1080)")
    live_parser.add_argument("--yes", "-y", action="store_true",
                             help="Skip confirmation prompt")

    sentiment_parser = subparsers.add_parser("sentiment", help="Check market sentiment")
    sentiment_parser.add_argument("--symbols", "-s", type=str, default="",
                                  help="Comma-separated symbols")

    subparsers.add_parser("pairs", help="List configured trading pairs")

    dashboard_parser = subparsers.add_parser("dashboard", help="Launch trading dashboard")
    dashboard_parser.add_argument("--port", type=int, default=8052, help="Dashboard port")
    dashboard_parser.add_argument("--mode", choices=["live", "paper", "offline"], default="paper",
                                   help="Dashboard mode")

    args = parser.parse_args()

    if args.command == "backtest":
        cmd_backtest(args)
    elif args.command == "paper":
        cmd_paper(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "sentiment":
        cmd_sentiment(args)
    elif args.command == "pairs":
        cmd_pairs(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
