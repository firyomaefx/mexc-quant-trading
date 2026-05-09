# Changelog

All notable changes to the MEXC Quant Trading Bot are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v3.0.0] — 2026-05-09

### Added
- 7-layer exit system (exits_v2): ATR trailing stop, Hurst regime exit, z-score velocity, partial profit, time stop
- LLM sentiment analyzer with GPT-4o-mini, Ollama, CryptoPanic news, keyword fallback
- PPO reinforcement learning meta-controller with 5-action space
- Dynamic Kelly position sizing with ensemble confidence scaling (stat 30%, ML 25%, sentiment 15%, MTF 15%, RL 15%)
- 60% total exposure cap
- Live paper trading engine with continuous MEXC data polling
- MEXC + CoinGecko hybrid connector with auto-failover
- Self-contained real-time Dash dashboard (dark theme, equity curve, trade log, risk status)
- Optuna Bayesian threshold optimization
- Walk-forward backtest with Monte Carlo and multi-pair portfolio simulation

### Fixed (DMAIC Audit)
- Dashboard proxy argument passed as wrong positional parameter
- `GoldConfig` import added to scalper.py — prevented runtime crash
- Equity-based position sizing (was always using initial capital, never compounding)
- Circuit breaker permanent halt logic (halt was immediately un-halting)
- Walk-forward backtest `bars_held` calculation (was always returning 1)
- Multi-pair backtest portfolio simulation (was no-op, equity never changed)
- RL agent `probs` variable initialization (was `UnboundLocalError`)
- Logistic regression `multi_class` parameter (was dead tautology)
- Threshold optimizer Sharpe ratio (was using +1/-1 proxies instead of real P&L std dev)
- MTF EMA slope normalized to percentage for cross-pair consistency

### Changed
- Time stop increased from 3 bars to 10 bars for crypto volatility
- Ensemble weights rebalanced to include RL as new layer
- Maximum total exposure capped at 60% of equity

### Breaking Changes
- `portfolio_risk.calculate_position_size()` signature updated — now requires `ml_conf`, `mtf_conf`, `sentiment`, `zscore_abs`, `rl_conf` parameters
- Exit logic migrated from `risk/exits.py` to `risk/exits_v2.py` — old module still used by V1/V2 gold code

---

## [v2.0.0] — 2026-04-XX

### Added
- Quant V2 trading framework with DOM-validated statistical trading
- Rithmic L2 order book integration
- 8 orderflow indicators
- 6-panel trading dashboard
- 18/18 test suite passing

---

## [v1.0.0] — 2026-03-XX

### Added
- Initial release: XAU/USD statistical trading system
- Mean reversion strategy with z-score and Hurst exponent
- 9/9 test suite passing
- Sharpe ratio 11.65

### Fixed
- Initial release — no prior fixes