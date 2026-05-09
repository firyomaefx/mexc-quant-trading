import sys

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config.crypto_config import CryptoConfig
from config.pairs import get_pair_config, list_enabled_symbols
from backtest.walk_forward import WalkForwardBacktester


class MultiPairBacktester:
    def __init__(self, config: CryptoConfig):
        self.config = config
        self.wfb = WalkForwardBacktester(config)
        self._pair_results: Dict[str, Dict] = {}
        self._portfolio_equity: Optional[np.ndarray] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict:
        self._pair_results = {}
        all_pair_trades = {}

        for symbol, df in data.items():
            res = self.wfb.run_pair(symbol, df,
                                    self.config.backtest.train_months,
                                    self.config.backtest.test_months)
            self._pair_results[symbol] = res

        for sym, res in self._pair_results.items():
            if "error" not in res:
                all_pair_trades[sym] = res.get("trades", [])

        combined_trades = []
        for sym in all_pair_trades:
            pair_trades = all_pair_trades[sym]
            if not pair_trades:
                pair_trades = []
            combined_trades.extend(pair_trades)

        portfolio = self._simulate_portfolio(data, all_pair_trades)
        correlations = self._compute_correlations(data)

        summary = self._build_summary(portfolio, correlations, combined_trades)
        return summary

    def _simulate_portfolio(self, data: Dict[str, pd.DataFrame],
                            pair_trades: Dict[str, List]) -> Dict:
        capital = self.config.backtest.initial_capital
        balance = capital
        equity_curve = [capital]
        peak = capital

        all_trades = []
        for sym, trades in pair_trades.items():
            all_trades.extend(trades)
        all_trades.sort(key=lambda t: t.get("entry_bar", 0) if isinstance(t, dict) else 0)

        for trade in all_trades:
            if isinstance(trade, dict) and "pnl" in trade:
                balance += trade["pnl"]
                equity_curve.append(balance)
                peak = max(peak, balance)

        if len(equity_curve) == 1:
            equity_curve.append(balance)

        return {
            "equity_curve": equity_curve,
            "final_equity": balance,
            "max_dd_pct": (peak - min(equity_curve)) / max(peak, 1) * 100,
        }

    def _compute_correlations(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if len(data) < 2:
            return pd.DataFrame()

        closes = {}
        for sym, df in data.items():
            if "close" in df.columns and len(df) > 10:
                closes[sym] = df["close"].values.astype(np.float64)

        if len(closes) < 2:
            return pd.DataFrame()

        returns = {}
        for sym, close in closes.items():
            ret = np.diff(close) / np.where(close[:-1] > 0, close[:-1], 1.0)
            returns[sym] = ret

        min_len = min(len(r) for r in returns.values())
        aligned = {sym: r[-min_len:] for sym, r in returns.items()}

        self._correlation_matrix = pd.DataFrame(aligned).corr()
        return self._correlation_matrix

    def _build_summary(self, portfolio: Dict, correlations: pd.DataFrame,
                       combined_trades: List[Dict]) -> Dict:
        total_return = 0.0
        n_wins = 0
        n_losses = 0
        total_pnl = 0.0

        for trades in combined_trades if combined_trades else []:
            if isinstance(trades, dict):
                pnl = trades.get("pnl", 0)
                if pnl > 0:
                    n_wins += 1
                else:
                    n_losses += 1
                total_pnl += pnl

        correlation_summary = {}
        if not correlations.empty and len(correlations.columns) >= 2:
            cols = correlations.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    pair_name = f"{cols[i]}_{cols[j]}"
                    correlation_summary[pair_name] = round(
                        float(correlations.iloc[i, j]), 3
                    )

        n_trades = n_wins + n_losses
        return {
            "pairs_tested": len(self._pair_results),
            "total_trades": n_trades,
            "win_rate": round(n_wins / n_trades * 100, 1) if n_trades > 0 else 0,
            "total_pnl": round(total_pnl, 2),
            "max_dd_pct": round(portfolio.get("max_dd_pct", 0), 2),
            "pair_correlations": correlation_summary,
            "pair_results": self._pair_results,
            "recommendation": self._generate_recommendation(),
        }

    def _generate_recommendation(self) -> str:
        valid_pairs = []
        for sym, res in self._pair_results.items():
            if "error" not in res:
                sharpe = res.get("mean_sharpe", -999)
                wr = res.get("mean_win_rate", 0)
                if sharpe > 0.5 and wr > 40:
                    valid_pairs.append((sym, sharpe))

        if not valid_pairs:
            return "No pair meets minimum criteria. Consider adjusting parameters."

        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = [p[0] for p in valid_pairs[:3]]

        return (f"Recommended trading pairs (highest Sharpe): {', '.join(top_pairs)}. "
                f"Trade max {self.config.scalping.max_concurrent_positions} concurrently "
                f"with strict risk management.")

    def print_summary(self, results: Dict):
        print("\n" + "=" * 70)
        print("  MULTI-PAIR PORTFOLIO BACKTEST")
        print("=" * 70)
        print(f"  Pairs tested: {results.get('pairs_tested', 0)}")
        print(f"  Total trades: {results.get('total_trades', 0)}")
        print(f"  Win rate: {results.get('win_rate', 0)}%")
        print(f"  Total PnL: ${results.get('total_pnl', 0):.2f}")
        print(f"  Max DD: {results.get('max_dd_pct', 0):.1f}%")

        corrs = results.get("pair_correlations", {})
        if corrs:
            print("\n  Pair Correlations:")
            for pair, corr in corrs.items():
                print(f"    {pair}: {corr:.3f}")

        print(f"\n  Recommendation: {results.get('recommendation', 'N/A')}")
        print("=" * 70)
