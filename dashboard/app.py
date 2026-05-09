import os
import sys
import argparse

from dotenv import load_dotenv
load_dotenv()

from dashboard.layout import app, build_full_layout
from dashboard.callbacks import register_callbacks
from dashboard.data_provider import DashboardDataProvider
from config.crypto_config import CryptoConfig


def create_dashboard(mode="paper", config=None, api_key="", api_secret=""):
    if config is None:
        config = CryptoConfig.with_default_pairs()

    if mode == "live":
        from live.scalper import Scalper
        scalper = Scalper(config, api_key=api_key, api_secret=api_secret)
        provider = DashboardDataProvider(config, scalper=scalper)
        provider.connected = getattr(scalper.mexc, "connected", False)
    elif mode == "paper":
        from live.paper_trader import PaperTrader
        proxy = os.getenv("MEXC_PROXY", "")
        from live.mexc_hybrid import MEXCHybridConnector
        mexc = MEXCHybridConnector(api_key, api_secret, proxy=proxy)
        mexc.connect()
        pt = PaperTrader(config, mexc)
        pt.run_simulation(list(config.scalping.pairs), max_bars=500)
        provider = DashboardDataProvider(config, paper_trader=pt)
        provider.connected = mexc.connected
    else:
        provider = DashboardDataProvider(config)

    app.layout = build_full_layout(config.scalping.pairs, 3000)
    register_callbacks(app, provider)
    return app, provider


def main():
    parser = argparse.ArgumentParser(description="Quant V2 Dashboard")
    parser.add_argument("--mode", choices=["live", "paper", "offline"], default="paper",
                        help="Dashboard mode")
    parser.add_argument("--port", type=int, default=8051, help="Dashboard port")
    args = parser.parse_args()

    config = CryptoConfig.with_default_pairs()
    api_key = os.getenv("MEXC_API_KEY", "")
    api_secret = os.getenv("MEXC_API_SECRET", "")

    print(f"\n{'='*55}")
    print(f"  QUANT V2 DASHBOARD - {args.mode.upper()} MODE")
    print(f"  Pairs: {config.scalping.pairs}")
    print(f"  Capital: ${config.scalping.initial_capital:.0f}")
    print(f"{'='*55}\n")

    app_instance, provider = create_dashboard(
        mode=args.mode, config=config,
        api_key=api_key, api_secret=api_secret,
    )

    print(f"  Opening dashboard at http://127.0.0.1:{args.port}")
    print(f"  Press Ctrl+C to stop\n")

    try:
        app_instance.run(debug=False, host="127.0.0.1", port=args.port)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        if getattr(provider, "scalper", None):
            provider.scalper.cleanup()


if __name__ == "__main__":
    main()