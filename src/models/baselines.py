"""The four baselines every result is reported against.

Absolute metrics mean almost nothing in this domain. A directional accuracy of
53 percent sounds like a result until you notice the fold's base rate was 53
percent. Only the delta against a baseline carries information, so these are
built first and evaluated on exactly the same rows as every model.

If a neural network does not clearly beat all four of these, there is no
finding, and the correct thing to do is say so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import CLASSIFICATION, REGRESSION, FoldData, Model

RET_LAG1 = "ret_lag1"
REALISED_VAR_NOW = "realised_var_now"


class ZeroBaseline(Model):
    """Predict no change.

    For a return target this forecasts zero, which is very close to the
    unconditional optimum for daily crypto. For direction it abstains at 0.5,
    which makes it a genuine null rather than a bet.
    """

    name = "zero"

    def fit(self, fold: FoldData) -> "ZeroBaseline":
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        if self.task == CLASSIFICATION:
            return np.full(len(X), 0.5)
        return np.zeros(len(X))

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(len(X), 0.5)


class PersistenceBaseline(Model):
    """Predict that tomorrow repeats today.

    The random walk in its most literal form. For direction it says up if the
    last observed return was positive. This is the baseline that most often
    embarrasses a complicated model.
    """

    name = "persistence"

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.sigma_ = 1e-2

    def fit(self, fold: FoldData) -> "PersistenceBaseline":
        if RET_LAG1 in fold.meta_train.columns:
            spread = float(fold.meta_train[RET_LAG1].std(skipna=True))
            self.sigma_ = spread if spread > 0 else 1e-2
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        last = _require(meta, RET_LAG1, len(X))
        if self.task == CLASSIFICATION:
            return (last > 0).astype(float)
        return last

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        """A graded probability rather than a hard 0 or 1.

        A hard label makes ROC-AUC and the calibration curve meaningless, and
        it also makes the baseline harder to compare against a model that does
        emit probabilities. Mapping the forecast through a normal CDF scaled by
        the training return standard deviation keeps the ranking identical while
        giving those metrics something to work with.
        """
        return _proba_from_forecast(_require(meta, RET_LAG1, len(X)), self.sigma_)


class HistoricalMeanBaseline(Model):
    """Predict the trailing mean return.

    A slightly more generous null than zero: it is allowed to notice that the
    sample has drifted upward. In a bull sample this is a meaningfully harder
    baseline to beat than zero, which is the point of including both.
    """

    name = "historical_mean"

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.window = int(self.params.get("historical_mean_window", 63))
        self.train_mean_ = 0.0
        self.sigma_ = 1e-2

    def fit(self, fold: FoldData) -> "HistoricalMeanBaseline":
        # The training mean is the fallback for test rows whose own trailing
        # window is unavailable. It is computed on train only.
        if RET_LAG1 in fold.meta_train.columns:
            self.train_mean_ = float(fold.meta_train[RET_LAG1].mean(skipna=True))
            spread = float(fold.meta_train[RET_LAG1].std(skipna=True))
            self.sigma_ = spread if spread > 0 else 1e-2
        self.fitted_ = True
        return self

    def _trailing(self, X: pd.DataFrame, meta: pd.DataFrame | None) -> np.ndarray:
        if meta is None or RET_LAG1 not in meta.columns:
            return np.full(len(X), self.train_mean_)
        trailing = (
            meta[RET_LAG1]
            .groupby(level="asset", observed=True)
            .transform(lambda s: s.rolling(self.window, min_periods=5).mean())
        )
        return trailing.fillna(self.train_mean_).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        values = self._trailing(X, meta)
        if self.task == CLASSIFICATION:
            return (values > 0).astype(float)
        return values

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        # Scaled by the standard error of a window mean rather than by the raw
        # return standard deviation, because a mean of `window` observations is
        # what is being compared against zero here.
        scale = self.sigma_ / np.sqrt(max(self.window, 1))
        return _proba_from_forecast(self._trailing(X, meta), scale)


class MajorityClassBaseline(Model):
    """Always predict the training fold's majority direction.

    Reported so that a classifier scoring exactly the base rate is immediately
    recognisable as having learned nothing.
    """

    name = "majority_class"

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.majority_ = 1.0
        self.base_rate_ = 0.5

    def fit(self, fold: FoldData) -> "MajorityClassBaseline":
        y = fold.y_train.dropna()
        self.base_rate_ = float(y.mean()) if len(y) else 0.5
        self.majority_ = 1.0 if self.base_rate_ >= 0.5 else 0.0
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        return np.full(len(X), self.majority_)

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        # Report the training base rate rather than a hard 0 or 1, so that the
        # probability is at least honest about its own confidence.
        return np.full(len(X), self.base_rate_)


class VolatilityPersistenceBaseline(Model):
    """Today's realised variance as the forecast for tomorrow's.

    The right null for the volatility target. Volatility clustering means this
    is a genuinely strong baseline, which is exactly why beating it with GARCH
    counts for something.
    """

    name = "vol_persistence"

    def __init__(self, params: dict | None = None, task: str = REGRESSION, seed: int = 42):
        super().__init__(params, task, seed)
        self.train_median_ = 0.0

    def fit(self, fold: FoldData) -> "VolatilityPersistenceBaseline":
        if REALISED_VAR_NOW in fold.meta_train.columns:
            values = np.log(fold.meta_train[REALISED_VAR_NOW].replace(0.0, np.nan).dropna())
            self.train_median_ = float(values.median()) if len(values) else 0.0
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        if meta is None or REALISED_VAR_NOW not in meta.columns:
            return np.full(len(X), self.train_median_)
        current = meta[REALISED_VAR_NOW].replace(0.0, np.nan)
        return np.log(current).fillna(self.train_median_).to_numpy(dtype=float)


class VolatilityClimatologyBaseline(Model):
    """Trailing mean log realised variance.

    The right unconditional null for a volatility target. Predicting a return
    baseline against a log variance target is not a weak forecast, it is a
    category error, so this replaces the historical mean baseline here.
    """

    name = "vol_climatology"

    def __init__(self, params: dict | None = None, task: str = REGRESSION, seed: int = 42):
        super().__init__(params, task, seed)
        self.window = int(self.params.get("historical_mean_window", 63))
        self.train_mean_ = 0.0

    def fit(self, fold: FoldData) -> "VolatilityClimatologyBaseline":
        if REALISED_VAR_NOW in fold.meta_train.columns:
            values = np.log(fold.meta_train[REALISED_VAR_NOW].replace(0.0, np.nan).dropna())
            self.train_mean_ = float(values.mean()) if len(values) else 0.0
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        if meta is None or REALISED_VAR_NOW not in meta.columns:
            return np.full(len(X), self.train_mean_)
        log_var = np.log(meta[REALISED_VAR_NOW].replace(0.0, np.nan))
        trailing = log_var.groupby(level="asset", observed=True).transform(
            lambda s: s.rolling(self.window, min_periods=10).mean()
        )
        return trailing.fillna(self.train_mean_).to_numpy(dtype=float)


BASELINE_REGISTRY = {
    "zero": ZeroBaseline,
    "persistence": PersistenceBaseline,
    "historical_mean": HistoricalMeanBaseline,
    "majority_class": MajorityClassBaseline,
    "vol_persistence": VolatilityPersistenceBaseline,
    "vol_climatology": VolatilityClimatologyBaseline,
}

# Which baselines make sense for which target. Majority class is meaningless
# for a regression target and volatility persistence is meaningless for
# direction, so neither is reported where it does not apply.
BASELINES_BY_TARGET = {
    "ret_1d": ["zero", "persistence", "historical_mean"],
    "dir_1d": ["zero", "persistence", "historical_mean", "majority_class"],
    "vol_1d": ["vol_persistence", "vol_climatology"],
}


def _proba_from_forecast(forecast: np.ndarray, sigma: float) -> np.ndarray:
    """Map a point forecast of a return to P(up) under a normal error model."""
    from scipy.stats import norm

    scale = sigma if sigma > 0 else 1e-6
    return norm.cdf(np.asarray(forecast, dtype=float) / scale)


def _require(meta: pd.DataFrame | None, column: str, n: int) -> np.ndarray:
    """Pull a raw column out of meta, falling back to zeros if it is absent."""
    if meta is None or column not in meta.columns:
        return np.zeros(n)
    return meta[column].fillna(0.0).to_numpy(dtype=float)


def build_baselines(target: str, params: dict | None = None, task: str = CLASSIFICATION) -> list[Model]:
    """Instantiate the baselines that apply to ``target``."""
    names = BASELINES_BY_TARGET.get(target, ["zero"])
    return [BASELINE_REGISTRY[name](params=params, task=task) for name in names]
