"""The interface every model family implements.

One interface for four very different families is only possible because they
all agree on the same input object: a feature matrix indexed by (asset, date),
sorted, with no gaps inside an asset. From that, a tree model reads rows, a
sequence model slices windows, and ARIMA pulls a single return column.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

REGRESSION = "regression"
CLASSIFICATION = "classification"

# Target name -> task type. Volatility is a regression on a log scale.
TARGET_TASKS = {
    "ret_1d": REGRESSION,
    "dir_1d": CLASSIFICATION,
    "vol_1d": REGRESSION,
}


@dataclass
class FoldData:
    """Everything one model needs for one walk-forward fold.

    ``X_*`` are feature matrices with a (asset, date) MultiIndex. ``y_*`` are
    Series sharing that index. ``meta_*`` carries columns a model must not
    train on but the evaluator needs, notably ``close`` and the realised
    forward return used by the backtest.
    """

    fold_id: int
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    meta_train: pd.DataFrame
    meta_test: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    target: str = "dir_1d"

    @property
    def task(self) -> str:
        return TARGET_TASKS[self.target]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]

    def describe(self) -> str:
        return (
            f"fold {self.fold_id}: train {self.train_start.date()}..{self.train_end.date()} "
            f"({len(self.X_train)} rows) test {self.test_start.date()}..{self.test_end.date()} "
            f"({len(self.X_test)} rows)"
        )


@dataclass
class Prediction:
    """A model's output on one fold.

    ``point`` is the forecast on the target's own scale. ``proba`` is only
    populated for classification and holds P(up).
    """

    index: pd.MultiIndex
    point: np.ndarray
    proba: np.ndarray | None = None
    model_name: str = ""
    fold_id: int = -1
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({"pred": self.point}, index=self.index)
        if self.proba is not None:
            frame["proba"] = self.proba
        frame["model"] = self.model_name
        frame["fold"] = self.fold_id
        if self.seed is not None:
            frame["seed"] = self.seed
        return frame


class Model(ABC):
    """Common fit/predict contract.

    Implementations must not look at test data during ``fit``. Anything that
    needs a validation signal uses ``fold.X_val`` / ``fold.y_val``, which the
    splitter carves out of the tail of the training window.
    """

    #: short identifier used in results tables and filenames
    name: str = "model"
    #: set True by families that ignore the scaled matrix and read raw columns
    wants_raw_features: bool = False
    #: set True by families whose output varies with the random seed
    is_stochastic: bool = False

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        self.params = dict(params or {})
        self.task = task
        self.seed = seed
        self.fitted_ = False

    @abstractmethod
    def fit(self, fold: FoldData) -> "Model":
        """Fit on ``fold.X_train`` / ``fold.y_train`` only."""

    @abstractmethod
    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        """Point forecast on the target's scale, aligned to ``X.index``.

        ``meta`` carries the raw columns the classical families work on
        directly, notably ``ret_lag1`` and ``realised_var_now``. It arrives
        already stripped of every target column, so a model cannot reach the
        answer through it even by accident.
        """

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray | None:
        """P(up) for classification models. None for everything else."""
        return None

    @staticmethod
    def _safe_meta(meta: pd.DataFrame | None) -> pd.DataFrame | None:
        """Drop every target column before a model is allowed to see meta."""
        if meta is None:
            return None
        leaky = [column for column in meta.columns if column in TARGET_TASKS or column.startswith("fwd_")]
        leaky += [column for column in meta.columns if column.startswith("realised_var_1d")]
        return meta.drop(columns=leaky, errors="ignore")

    def run_fold(self, fold: FoldData) -> Prediction:
        """Fit and predict in one call. The standard path used by the runner."""
        self.fit(fold)
        meta = self._safe_meta(fold.meta_test)
        point = np.asarray(self.predict(fold.X_test, meta), dtype=float)
        proba = self.predict_proba(fold.X_test, meta) if fold.task == CLASSIFICATION else None
        if proba is not None:
            proba = np.asarray(proba, dtype=float)
        return Prediction(
            index=fold.X_test.index,
            point=point,
            proba=proba,
            model_name=self.name,
            fold_id=fold.fold_id,
            seed=self.seed if self.is_stochastic else None,
        )

    def feature_importance(self) -> pd.Series | None:
        """Optional. Returns a Series indexed by feature name."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, task={self.task!r})"
