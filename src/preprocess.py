"""Per-fold feature preprocessing.

Constraint C5 of the project spec says every scaler, imputer and feature
statistic is fit on the training fold only. Fitting a ``StandardScaler`` on the
whole series leaks future volatility backward into the past, and it is the most
common bug in this category of project.

Centralising it here rather than leaving it to each model means there is one
place to audit and one object for ``tests/test_no_leakage.py`` to interrogate.
The preprocessor is constructed fresh inside every fold; it is never reused
across folds and it never sees test data during ``fit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FoldPreprocessor:
    """Winsorise then standardise, with every statistic learned on train only.

    Attributes are populated by :meth:`fit` and are exposed deliberately so a
    test can assert they came from the training rows and nowhere else.
    """

    winsorize: bool = True
    lower_q: float = 0.01
    upper_q: float = 0.99
    scale: str = "standard"  # "standard" | "none"

    lower_bounds_: pd.Series | None = field(default=None, init=False)
    upper_bounds_: pd.Series | None = field(default=None, init=False)
    means_: pd.Series | None = field(default=None, init=False)
    stds_: pd.Series | None = field(default=None, init=False)
    medians_: pd.Series | None = field(default=None, init=False)
    columns_: list[str] | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)
    n_fit_rows_: int = field(default=0, init=False)

    def fit(self, X_train: pd.DataFrame) -> "FoldPreprocessor":
        """Learn clip bounds, centring and scaling from the training rows."""
        if X_train.empty:
            raise ValueError("cannot fit a preprocessor on an empty training frame")

        self.columns_ = list(X_train.columns)
        self.n_fit_rows_ = len(X_train)

        if self.winsorize:
            self.lower_bounds_ = X_train.quantile(self.lower_q)
            self.upper_bounds_ = X_train.quantile(self.upper_q)
            clipped = X_train.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        else:
            self.lower_bounds_ = None
            self.upper_bounds_ = None
            clipped = X_train

        # Imputation values also come from the training fold. A feature that is
        # entirely NaN in train gets a median of NaN, so it is filled with zero
        # after standardisation, which is the neutral value.
        self.medians_ = clipped.median()

        if self.scale == "standard":
            self.means_ = clipped.mean()
            std = clipped.std(ddof=0)
            # A constant column has zero variance. Dividing by it produces inf,
            # so it is left unscaled and ends up as a column of zeros.
            self.stds_ = std.replace(0.0, 1.0).fillna(1.0)
        else:
            self.means_ = None
            self.stds_ = None

        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned bounds and statistics. Never re-estimates anything."""
        if not self.fitted_:
            raise RuntimeError("FoldPreprocessor.transform called before fit")

        missing = set(self.columns_) - set(X.columns)
        if missing:
            raise ValueError(f"transform is missing columns learned at fit time: {sorted(missing)}")

        out = X[self.columns_].copy()

        if self.winsorize:
            out = out.clip(self.lower_bounds_, self.upper_bounds_, axis=1)

        out = out.fillna(self.medians_)

        if self.scale == "standard":
            out = (out - self.means_) / self.stds_

        # Anything still not finite after all of that becomes the neutral value.
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X_train).transform(X_train)

    @classmethod
    def from_config(cls, config: dict) -> "FoldPreprocessor":
        features = config.get("features", {})
        winsorize = features.get("winsorize", {})
        return cls(
            winsorize=winsorize.get("enabled", True),
            lower_q=winsorize.get("lower_q", 0.01),
            upper_q=winsorize.get("upper_q", 0.99),
            scale=features.get("scale", "standard"),
        )
