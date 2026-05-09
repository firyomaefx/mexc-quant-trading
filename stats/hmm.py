import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:
    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=200,
            random_state=random_state,
        )
        self._fitted = False
        self._scaler = StandardScaler()
        self._state_labels = {}
        self._ranging_state = None

    def fit(self, returns: np.ndarray) -> "HMMRegimeDetector":
        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[np.isfinite(returns)]

        if len(returns) < 100:
            raise ValueError(f"Need at least 100 observations, got {len(returns)}")

        log_returns = np.log(1 + np.clip(returns, -0.99, None))

        features = np.column_stack([
            log_returns,
            np.abs(log_returns),
            np.abs(log_returns) ** 2,
        ])

        scaled = self._scaler.fit_transform(features)
        self.model.fit(scaled)
        self._fitted = True

        states = self.model.predict(scaled)
        state_vols = {}
        for s in range(self.n_states):
            state_vols[s] = np.std(returns[states == s])

        self._ranging_state = min(state_vols, key=state_vols.get)

        label_order = sorted(state_vols, key=state_vols.get)
        self._state_labels = {label_order[0]: "ranging", label_order[-1]: "trending"}
        if self.n_states > 2:
            for i in range(1, self.n_states - 1):
                self._state_labels[label_order[i]] = f"regime_{i}"

        return self

    def predict_state(self, returns_recent: np.ndarray) -> tuple:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")

        returns_recent = np.asarray(returns_recent, dtype=np.float64).ravel()

        log_returns = np.log(1 + np.clip(returns_recent, -0.99, None))
        features = np.column_stack([
            log_returns,
            np.abs(log_returns),
            np.abs(log_returns) ** 2,
        ])
        scaled = self._scaler.transform(features)

        state = self.model.predict(scaled)[-1]
        probs = self.model.predict_proba(scaled)[-1]

        return state, probs

    def is_ranging(self, prob_threshold: float = 0.80) -> bool:
        if not self._fitted or self._ranging_state is None:
            return False

        if not hasattr(self, "_latest_state"):
            return False

        probs = getattr(self, "_latest_probs", None)
        if probs is None:
            return False

        return probs[self._ranging_state] >= prob_threshold

    def update(self, returns_recent: np.ndarray):
        state, probs = self.predict_state(returns_recent)
        self._latest_state = state
        self._latest_probs = probs

    def predict_proba_series(self, returns: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predicting")

        returns = np.asarray(returns, dtype=np.float64).ravel()
        log_returns = np.log(1 + np.clip(returns, -0.99, None))
        features = np.column_stack([
            log_returns,
            np.abs(log_returns),
            np.abs(log_returns) ** 2,
        ])
        scaled = self._scaler.transform(features)
        probs = self.model.predict_proba(scaled)

        if self._ranging_state is not None:
            return probs[:, self._ranging_state]

        return np.full(len(returns), 0.5)

    def get_ranging_probability(self) -> float:
        if not hasattr(self, "_latest_probs") or self._ranging_state is None:
            return 0.0
        return float(self._latest_probs[self._ranging_state])

    def get_state_label(self, state: int) -> str:
        return self._state_labels.get(state, f"unknown_{state}")
