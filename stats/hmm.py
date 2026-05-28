import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    HAS_HMM = False


class HMMRegimeDetector:
    def __init__(self, n_states: int = 2, random_state: int = 42):
        self.n_states = n_states
        self.random_state = random_state
        self._fitted = False
        self._ranging_state = None
        self._state_labels = {}
        self._latest_state = 0
        self._latest_probs = None

        if HAS_HMM:
            self.model = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=200,
                random_state=random_state,
            )
            self._scaler = StandardScaler()
        else:
            self.model = None
            self._scaler = None

    def fit(self, returns: np.ndarray) -> "HMMRegimeDetector":
        if not HAS_HMM or self.model is None:
            return self

        returns = np.asarray(returns, dtype=np.float64).ravel()
        returns = returns[np.isfinite(returns)]

        if len(returns) < 100:
            return self

        log_returns = np.log(1 + np.clip(returns, -0.99, None))
        features = np.column_stack([log_returns, np.abs(log_returns), np.abs(log_returns) ** 2])
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
        return self

    def predict_state(self, returns_recent: np.ndarray) -> tuple:
        if not self._fitted or self.model is None:
            return 0, np.array([0.5, 0.5] + [0.0] * (self.n_states - 2))

        returns_recent = np.asarray(returns_recent, dtype=np.float64).ravel()
        log_returns = np.log(1 + np.clip(returns_recent, -0.99, None))
        features = np.column_stack([log_returns, np.abs(log_returns), np.abs(log_returns) ** 2])
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
        if not self._fitted or self.model is None:
            return np.full(len(returns), 0.5)
        returns = np.asarray(returns, dtype=np.float64).ravel()
        log_returns = np.log(1 + np.clip(returns, -0.99, None))
        features = np.column_stack([log_returns, np.abs(log_returns), np.abs(log_returns) ** 2])
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