# MEXC Quant Trading Bot v3.0

> AI-powered multi-pair crypto scalping bot — ML ensemble, LLM sentiment, RL agent, dynamic Kelly sizing, 7-layer exit system, live dashboard.

![Python](https://img.shields.io/badge/Python-3.14-blue)
![ccxt](https://img.shields.io/badge/ccxt-4.5.50-orange)
![Dash](https://img.shields.io/badge/Dash-2.18+-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

AI-driven automated trading system built for MEXC exchange. Combines statistical mean-reversion (z-score/Hurst), machine learning (XGBoost+LR+k-NN ensemble), LLM sentiment analysis (GPT-4o-mini/Ollama), reinforcement learning (PPO), multi-timeframe ADX filtering, and dynamic Kelly position sizing. Runs on spot or futures with 3-5x leverage, targeting $0.50-$1 per trade on $160 capital.

## Architecture

```
                    ┌─────────────────────────────┐
                    │       main_mexc.py (CLI)     │
                    │  backtest│paper│live│dashboard│
                    └──────────┬──────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
     ┌──────▼──────┐   ┌──────▼──────┐   ┌──────▼──────┐
     │   Signals   │   │    Risk     │   │  Execution  │
     │             │   │             │   │             │
     │ z-score     │   │ exits_v2    │   │ scalper.py  │
     │ Hurst       │   │ circuit_brk │   │ paper_trade │
     │ ML ensemble │   │ portfolio   │   │ live_paper  │
     │ LLM sentim  │   │ futures_risk│   │ mexc_hybrid │
     │ RL agent    │   │             │   │             │
     │ MTF ADX     │   │             │   │             │
     │ Correlation │   │             │   │             │
     └─────────────┘   └─────────────┘   └─────────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │       Dashboard (Dash)       │
                    │  Portfolio│Pairs│Risk│Charts │
                    └──────────────────────────────┘
```

## 7-Layer Exit System (exits_v2)

| # | Exit Type | Trigger |
|---|-----------|---------|
| 1 | Catastrophic z-stop | z-score breaches configurable threshold |
| 2 | ATR trailing stop | Price reverses > 1.5× ATR from extreme |
| 3 | Hurst regime exit | Hurst > 0.55 signals trending (exit mean-revert) |
| 4 | Z-score velocity exit | Rapid z-score decay against position |
| 5 | Partial profit | Close 50% at z = ±0.5 |
| 6 | Z-score mean target | Full exit at z = 0.0 |
| 7 | Time stop | Exit after 10 bars regardless |

## Features

- **Multi-pair**: XRP/USDT, ADA/USDT, SOL/USDT + DOGE, LTC, AVAX (configurable)
- **ML ensemble**: XGBoost + LogisticRegression + k-NN majority voting, auto-retrain every 50 trades
- **LLM sentiment**: GPT-4o-mini or Ollama local models, CryptoPanic news aggregation, keyword fallback
- **RL meta-controller**: PPO with 5 actions (hold/long/short/close_long/close_short), auto-retrain when win rate < 40%
- **Dynamic Kelly sizing**: `equity × risk% × ensemble_confidence / (ATR × mult)`, 60% total exposure cap
- **7-layer circuit breaker**: spread gate, z-confidence, consecutive loss, daily -5%, DD -15%, cooldown, volatility
- **Walk-forward backtest**: Monte Carlo 1000 runs, Optuna Bayesian parameter optimization
- **Real-time dashboard**: Dark theme, equity curve, trade log, per-pair signals, risk status
- **MEXC + CoinGecko failover**: Auto-detects MEXC connectivity, falls back to CoinGecko free API
- **Spot + Futures**: 3-5x leverage with liquidation distance monitoring

## Quickstart

```bash
git clone https://github.com/firyomaefx/mexc-quant-trading.git
cd mexc-quant-trading
pip install -r requirements.txt
cp ../.env.example .env  # edit with your MEXC keys
python main_mexc.py pairs        # list configured pairs
python main_mexc.py sentiment    # check market sentiment
python main_mexc.py paper -b 500 # paper trade 500 bars
python main_mexc.py dashboard --mode paper  # launch dashboard
```

## Commands

| Command | Description |
|---------|-------------|
| `python main_mexc.py backtest` | Walk-forward backtest with Monte Carlo |
| `python main_mexc.py paper -b 2000` | Paper trading simulation (2000 bars) |
| `python main_mexc.py live --yes` | Start live trading on MEXC |
| `python main_mexc.py dashboard --mode paper` | Launch real-time dashboard (port 8052) |
| `python main_mexc.py sentiment` | Check crypto market sentiment |
| `python main_mexc.py pairs` | List configured trading pairs |

## Configuration (.env)

| Key | Description | Required |
|-----|-------------|----------|
| `MEXC_API_KEY` | MEXC exchange API key | For live trading |
| `MEXC_API_SECRET` | MEXC exchange API secret | For live trading |
| `MEXC_PROXY` | SOCKS5/HTTP proxy (if MEXC blocked) | No |
| `OPENAI_API_KEY` | OpenAI key for LLM sentiment | No |
| `OLLAMA_HOST` | Ollama local server address | No |

## DMAIC Audit (v3.0.0)

10 critical and high-priority issues identified and resolved:
1. Dashboard proxy arg passed as wrong positional
2. `GoldConfig` import added, equity-based sizing
3. Circuit breaker permanent halt now correctly permanent
4. Walk-forward `bars_held` calculation (was always 1)
5. Multi-pair backtest portfolio simulation (was no-op)
6. RL agent `probs` unbound variable
7. LR `multi_class` tautology (now correct for >2 classes)
8. Threshold optimizer Sharpe using real P&L std dev
9. MTF EMA slope normalized to percentage for cross-pair consistency
10. `close_position` added to hybrid connector for futures path

## Project Structure

```
quant_v2/
├── main_mexc.py                # CLI entry point
├── config/
│   ├── crypto_config.py        # All configuration dataclasses
│   └── pairs.py                # Pair definitions and defaults
├── signals/
│   ├── llm_sentiment.py        # GPT-4o-mini / Ollama sentiment
│   ├── ml_ensemble.py          # XGBoost+LR+k-NN ensemble
│   ├── mtf_filter.py           # Multi-timeframe ADX + trend
│   ├── pair_correlation.py     # Cross-pair correlation block
│   ├── rl_agent.py             # PPO reinforcement learning
│   └── sentiment.py            # Keyword-based sentiment fallback
├── risk/
│   ├── circuit_breaker.py      # 7-layer risk guard
│   ├── exits_v2.py             # 7-exit-type system
│   ├── futures_risk.py         # Liquidation distance monitor
│   └── portfolio_risk.py       # Dynamic Kelly + ensemble sizing
├── live/
│   ├── scalper.py              # Async live execution loop
│   ├── paper_trader.py         # Paper trading simulation
│   ├── live_paper.py           # Continuous live paper engine
│   ├── mexc_adapter.py         # ccxt MEXC spot adapter
│   ├── mexc_futures.py         # ccxt MEXC futures adapter
│   ├── mexc_hybrid.py          # MEXC + CoinGecko fallback
│   └── mexc_ws.py              # WebSocket streams
├── backtest/
│   ├── walk_forward.py         # Walk-forward + Monte Carlo
│   └── multi_pair_bt.py        # Portfolio-level combined backtest
├── training/
│   ├── trading_env.py          # OpenAI Gym trading environment
│   └── threshold_optimizer.py  # Optuna Bayesian optimizer
├── dashboard/
│   ├── run.py                  # Self-contained live dashboard
│   ├── layout.py               # Dark theme UI components
│   ├── callbacks.py            # Dash reactive callbacks
│   └── data_provider.py        # Real-time data bridge
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## License

MIT — see [LICENSE](./LICENSE) for details.

## Changelog

See [CHANGELOG.md](./CHANGELOG.md)