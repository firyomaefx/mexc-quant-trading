import os
import sys
import sys

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class PairConfig:
    symbol: str
    base: str
    quote: str
    min_qty: float
    qty_step: float
    min_notional: float
    pip_value: float
    typical_spread_pct: float
    zscore_entry_long: float = -1.8
    zscore_entry_short: float = 2.0
    zscore_stop_long: float = -2.8
    zscore_stop_short: float = 2.8
    time_stop_bars: int = 10
    atr_multiplier_sl: float = 1.5
    enabled: bool = True


@dataclass
class ExitConfig:
    time_stop_bars: int = 10
    atr_trailing_mult: float = 1.5
    partial_profit_target_z: float = 0.5
    partial_profit_pct: float = 0.5
    hurst_exit_threshold: float = 0.55
    zscore_velocity_threshold: float = 0.3
    require_velocity_flat: bool = False


@dataclass
class ScalpingConfig:
    pairs: List[str] = field(default_factory=lambda: ["XRP/USDT", "ADA/USDT", "SOL/USDT"])
    primary_tf: int = 1
    secondary_tf: int = 5
    tertiary_tf: int = 15
    initial_capital: float = 160.0
    account_risk_pct: float = 0.015
    max_daily_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_concurrent_positions: int = 3
    max_positions_per_pair: int = 1
    cooldown_seconds: float = 90.0
    min_zscore_confidence: float = 1.2
    max_spread_pct: float = 0.15
    max_volatility_ratio: float = 2.0
    max_consecutive_losses: int = 3
    trade_min_interval_seconds: float = 60.0
    max_total_exposure_pct: float = 0.60
    dynamic_sizing: bool = True


@dataclass
class MLConfig:
    enabled: bool = True
    prob_threshold: float = 0.55
    adaptive_window: int = 100
    ensemble_weight_stat: float = 0.30
    ensemble_weight_ml: float = 0.25
    ensemble_weight_sentiment: float = 0.15
    ensemble_weight_mtf: float = 0.15
    ensemble_weight_rl: float = 0.15
    min_trades_for_ml: int = 100
    retrain_interval: int = 50
    features: List[str] = field(default_factory=lambda: [
        "zscore", "hurst", "velocity", "atr_ratio", "spread_pct",
        "volume_ratio", "hour_sin", "hour_cos", "rsi_14",
        "hurst_x_zscore", "sma_ratio",
    ])


@dataclass
class LLMSentimentConfig:
    enabled: bool = True
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    cache_seconds: int = 600
    max_headlines: int = 10
    fallback_to_keywords: bool = True


@dataclass
class RLConfig:
    enabled: bool = False
    state_dim: int = 10
    lookback: int = 50
    train_timesteps: int = 50000
    retrain_interval: int = 200
    model_path: str = "quant_v2/training/rl_model"


@dataclass
class SentimentConfig:
    enabled: bool = True
    fear_greed_weight: float = 0.40
    news_weight: float = 0.30
    price_action_weight: float = 0.30
    cache_seconds: int = 600
    fear_greed_url: str = "https://api.alternative.me/fng/"
    cryptopanic_api_key: str = ""
    cryptopanic_url: str = "https://cryptopanic.com/api/v1/posts/"


@dataclass
class FuturesConfig:
    enabled: bool = False
    max_leverage: int = 3
    margin_mode: str = "isolated"
    liq_safety_pct: float = 0.20
    funding_rate_limit: float = 0.01
    min_margin_ratio: float = 0.15


@dataclass
class PaperConfig:
    latency_ms: int = 300
    slippage_pct: float = 0.08
    min_trades_for_live: int = 100
    min_win_rate_for_live: float = 0.50
    min_expectancy_for_live: float = 0.0
    report_dir: str = "paper_reports"


@dataclass
class BacktestConfig:
    initial_capital: float = 160.0
    commission_pct: float = 0.05
    slippage_pct: float = 0.08
    train_months: int = 3
    test_months: int = 1
    monte_carlo_runs: int = 1000
    min_required_trades: int = 30
    n_optuna_trials: int = 100


@dataclass
class CryptoConfig:
    pairs: Dict[str, PairConfig] = field(default_factory=dict)
    scalping: ScalpingConfig = field(default_factory=ScalpingConfig)
    exit: ExitConfig = field(default_factory=ExitConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    llm_sentiment: LLMSentimentConfig = field(default_factory=LLMSentimentConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    futures: FuturesConfig = field(default_factory=FuturesConfig)
    paper: PaperConfig = field(default_factory=PaperConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)

    @classmethod
    def with_default_pairs(cls) -> "CryptoConfig":
        from config.pairs import DEFAULT_PAIRS
        cfg = cls()
        cfg.pairs = DEFAULT_PAIRS
        return cfg


CRYPTO_CONFIG = CryptoConfig.with_default_pairs()