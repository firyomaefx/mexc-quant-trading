import os
import sys
import sys

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from collections import deque

from config.crypto_config import MLConfig


class MLSignalEnhancer:
    def __init__(self, config: MLConfig = None):
        self.config = config or MLConfig()
        self._models: Dict[str, object] = {}
        self._scaler = None
        self._fitted = False
        self._trade_history: deque = deque(maxlen=self.config.adaptive_window)
        self._feature_means: Dict[str, float] = {}
        self._feature_stds: Dict[str, float] = {}
        self._feature_cols: List[str] = []
        self._retrain_counter = 0
        self._last_retrain_trades = 0
        self._retrain_interval = 50

    def fit(self, features_df: pd.DataFrame, labels: np.ndarray):
        from sklearn.preprocessing import StandardScaler

        feature_cols = [c for c in self.config.features if c in features_df.columns]
        if len(feature_cols) < 3:
            return False

        X = features_df[feature_cols].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)

        if len(X) < self.config.min_trades_for_ml:
            return False

        mask = np.isfinite(labels)
        X = X.loc[mask]
        labels = labels[mask]

        if len(X) < self.config.min_trades_for_ml:
            return False

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._feature_cols = feature_cols

        for i, col in enumerate(feature_cols):
            self._feature_means[col] = float(self._scaler.mean_[i])
            self._feature_stds[col] = float(self._scaler.scale_[i])

        unique_labels = np.unique(labels)
        n_classes = len(unique_labels)

        if n_classes == 1:
            return False

        try:
            from xgboost import XGBClassifier
            n_est = min(200, max(50, len(X) // 5))
            self._models["xgb"] = XGBClassifier(
                n_estimators=n_est, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42, use_label_encoder=False,
            )
            self._models["xgb"].fit(X_scaled, labels)
        except Exception:
            self._models.pop("xgb", None)

        try:
            from sklearn.linear_model import LogisticRegression
            self._models["lr"] = LogisticRegression(
                max_iter=1000, C=0.5, random_state=42,
                multi_class="auto" if n_classes <= 2 else "multinomial",
            )
            self._models["lr"].fit(X_scaled, labels)
        except Exception:
            self._models.pop("lr", None)

        try:
            from sklearn.neighbors import KNeighborsClassifier
            n_neighbors = min(7, max(3, len(X) // 20))
            self._models["knn"] = KNeighborsClassifier(
                n_neighbors=n_neighbors, weights="distance",
            )
            self._models["knn"].fit(X_scaled, labels)
        except Exception:
            self._models.pop("knn", None)

        if len(self._models) > 0:
            self._fitted = True
            print(f"ML ensemble fitted: {len(self._models)} models on {len(X)} samples, "
                  f"{len(feature_cols)} features, {n_classes} classes")
            return True
        return False

    def predict_ensemble(self, features: Dict[str, float]) -> Optional[float]:
        if not self._fitted or not self._models:
            return None

        feature_cols = [c for c in self._feature_cols if c in self._feature_means]
        if len(feature_cols) < 3:
            return None

        X = np.zeros(len(feature_cols))
        for i, col in enumerate(feature_cols):
            raw = features.get(col, 0.0)
            mean = self._feature_means.get(col, 0.0)
            std = self._feature_stds.get(col, 1.0)
            X[i] = (raw - mean) / std if std > 1e-10 else 0.0

        weights = {"xgb": 0.50, "lr": 0.25, "knn": 0.25}
        probas = []

        for name, model in self._models.items():
            w = weights.get(name, 0.25)
            try:
                pred = model.predict_proba(X.reshape(1, -1))
                if pred is not None and pred.shape[1] >= 2:
                    probas.append((w, float(pred[0][1])))
            except Exception:
                pass

        if not probas:
            return None

        total_weight = sum(w for w, _ in probas)
        weighted_prob = sum(w * p for w, p in probas) / total_weight if total_weight > 0 else 0.5
        return float(np.clip(weighted_prob, 0.0, 1.0))

    def should_trade(self, features: Dict[str, float], base_signal: int) -> Tuple[bool, float, str]:
        if not self.config.enabled or not self._fitted:
            return True, 1.0, "ml_disabled"

        prob = self.predict_ensemble(features)
        if prob is None:
            return True, 1.0, "ml_no_prediction"

        confidence = prob if base_signal == 1 else (1.0 - prob)

        if confidence >= self.config.prob_threshold:
            return True, confidence, "ml_confirmed"
        else:
            return False, confidence, "ml_rejected"

    def update_trade_history(self, pnl: float, features: Dict[str, float], signal: int):
        self._trade_history.append({
            "pnl": pnl,
            "features": features.copy() if features else {},
            "signal": signal,
            "timestamp": pd.Timestamp.now(),
        })
        self._retrain_counter += 1

    def maybe_retrain(self, features_df: pd.DataFrame = None) -> bool:
        if self._retrain_counter - self._last_retrain_trades < self._retrain_interval:
            return False
        if len(self._trade_history) < self.config.min_trades_for_ml:
            return False

        trades = list(self._trade_history)
        labels = np.array([1 if t["pnl"] > 0 else 0 for t in trades])

        if features_df is not None and len(features_df) >= len(trades):
            recent = features_df.iloc[-len(trades):]
            try:
                if self.fit(recent, labels):
                    self._last_retrain_trades = self._retrain_counter
                    print(f"ML retrained at {self._retrain_counter} trades")
                    return True
            except Exception as e:
                print(f"ML retrain failed: {e}")

        if len(self._trade_history) >= 100:
            self.adapt_thresholds()

        return False

    def adapt_thresholds(self) -> Dict[str, float]:
        if len(self._trade_history) < 20:
            return {"prob_threshold": self.config.prob_threshold}

        recent = list(self._trade_history)[-100:]
        wins = [t for t in recent if t["pnl"] > 0]
        win_rate = len(wins) / len(recent)

        if win_rate < 0.40:
            new_threshold = min(0.80, self.config.prob_threshold + 0.10)
        elif win_rate < 0.50:
            new_threshold = min(0.70, self.config.prob_threshold + 0.05)
        elif win_rate > 0.60:
            new_threshold = max(0.40, self.config.prob_threshold - 0.05)
        elif win_rate > 0.65:
            new_threshold = max(0.35, self.config.prob_threshold - 0.10)
        else:
            new_threshold = self.config.prob_threshold

        old_threshold = self.config.prob_threshold
        self.config.prob_threshold = new_threshold
        return {
            "prob_threshold": new_threshold,
            "old_threshold": old_threshold,
            "recent_win_rate": win_rate,
            "recent_trades": len(recent),
        }

    @staticmethod
    def prepare_labels_3class(signals_df: pd.DataFrame, forward_bars: int = 5,
                              small_win_pct: float = 0.003,
                              big_win_pct: float = 0.01) -> np.ndarray:
        close = signals_df["close"].values if "close" in signals_df.columns else None
        if close is None:
            return np.zeros(len(signals_df), dtype=int)

        labels = np.zeros(len(signals_df), dtype=int)
        for i in range(len(signals_df) - forward_bars):
            if signals_df["signal"].iloc[i] == 0:
                labels[i] = 0
                continue

            future_close = close[i + forward_bars]
            current_close = close[i]
            if current_close <= 0:
                continue

            pct_change = (future_close - current_close) / current_close
            direction = signals_df["signal"].iloc[i]

            realized = pct_change * direction

            if realized >= big_win_pct:
                labels[i] = 2
            elif realized >= small_win_pct:
                labels[i] = 1
            elif realized <= -big_win_pct:
                labels[i] = 0
            else:
                labels[i] = 0

        return labels

    @staticmethod
    def prepare_labels_from_signals(signals_df: pd.DataFrame, forward_bars: int = 5,
                                    min_profit_pct: float = 0.003) -> np.ndarray:
        return MLSignalEnhancer.prepare_labels_3class(
            signals_df, forward_bars, min_profit_pct, min_profit_pct * 3
        )

    @staticmethod
    def compute_ml_features(df: pd.DataFrame, spread_pct: float = 0.05) -> pd.DataFrame:
        features = pd.DataFrame(index=df.index)

        if "zscore" in df.columns:
            features["zscore"] = df["zscore"]
        if "hurst" in df.columns:
            features["hurst"] = df["hurst"]
        if "velocity" in df.columns:
            features["velocity"] = df["velocity"].fillna(0)

        close = df["close"].values.astype(np.float64) if "close" in df.columns else np.ones(len(df))
        features["atr_ratio"] = MLSignalEnhancer._atr_ratio(close, period=14)

        features["spread_pct"] = spread_pct

        if "volume" in df.columns:
            vol = df["volume"].values.astype(np.float64)
            sma_vol = pd.Series(vol).rolling(20).mean().values
            features["volume_ratio"] = np.where(sma_vol > 0, vol / sma_vol, 1.0)
        else:
            features["volume_ratio"] = 1.0

        if df.index is not None and hasattr(df.index, "hour"):
            hours = np.array([t.hour for t in df.index], dtype=np.float64)
        else:
            hours = np.zeros(len(df))
        features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
        features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

        rsi = MLSignalEnhancer._compute_rsi(close, period=14)
        features["rsi_14"] = np.where(np.isfinite(rsi), rsi / 100.0, 0.5)

        if "hurst" in df.columns:
            features["hurst_x_zscore"] = df["hurst"].values * np.abs(df["zscore"].values if "zscore" in df.columns else np.ones(len(df)))
        else:
            features["hurst_x_zscore"] = 0.0

        if len(close) > 20:
            sma5 = pd.Series(close).rolling(5).mean().values
            sma20 = pd.Series(close).rolling(20).mean().values
            features["sma_ratio"] = np.where(sma20 > 0, sma5 / sma20, 1.0)
        else:
            features["sma_ratio"] = 1.0

        return features.fillna(0).replace([np.inf, -np.inf], 0)

    @staticmethod
    def _atr_ratio(close: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(close)
        if n < period + 1:
            return np.ones(n)

        tr = np.abs(np.diff(close, prepend=close[0]))
        atr = np.zeros(n)
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

        sma = pd.Series(close).rolling(period).mean().values
        ratio = np.where((sma > 0) & (atr > 0), atr / sma, 0.01)
        return ratio

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
        n = len(close)
        if n < period + 1:
            return np.full(n, 50.0)

        deltas = np.diff(close, prepend=close[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        avg_gain[period] = np.mean(gains[1:period + 1])
        avg_loss[period] = np.mean(losses[1:period + 1])

        for i in range(period + 1, n):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

        rs = np.where(avg_loss > 0, np.where(np.isfinite(avg_gain / avg_loss), avg_gain / avg_loss, 100.0), 100.0)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi