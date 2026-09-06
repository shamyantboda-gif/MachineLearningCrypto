"""L2-penalised linear regression on the tabular feature matrix.

This is the control the tree and sequence models have to beat before their
complexity means anything. The feature matrix reaching a model has already been
winsorised and standardised on the training fold alone, which is exactly the
input a ridge wants: bounded, centred, on a common scale. If a hundred-odd
engineered columns carry any linear signal about tomorrow, a penalised linear
model finds it, and it does so with one hyper-parameter and coefficients you can
read off directly.

The penalty is not a formality. With more features than usable independent
observations in a fold and a signal to noise ratio near zero, an unpenalised
ordinary least squares fit will happily allocate enormous offsetting weights to
correlated columns and produce forecasts that swing far outside the range of
anything it trained on. Shrinkage is what keeps the fit inside the data, which
is why ``alpha`` is selected per fold rather than fixed once.

On a classification target the family is L2-penalised logistic regression. That
is the same ridge penalty applied to the log-odds, and it matters that it is
logistic rather than a ``RidgeClassifier``: the headline comparison in this
project runs Diebold-Mariano on Brier scores, which needs a genuine probability,
not a point forecast pushed through an assumed error distribution.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import CLASSIFICATION, FoldData, Model

#: Shrinkage grid searched on the inner validation slice. Spans four decades
#: because the right amount of shrinkage depends on how much of the fold the
#: expanding window has accumulated, and that changes across the walk forward.
DEFAULT_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]


class RidgeModel(Model):
    """Ridge regression, or L2 logistic regression, with alpha picked per fold."""

    name = "ridge"
    is_stochastic = False

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.alphas = [float(a) for a in self.params.get("alphas", DEFAULT_ALPHAS)]
        self.default_alpha = float(self.params.get("default_alpha", 10.0))
        self.fit_intercept = bool(self.params.get("fit_intercept", True))
        self.max_iter = int(self.params.get("max_iter", 1000))

        self.estimator_ = None
        self.feature_names_: list[str] = []
        self.best_alpha_: float | None = None
        self.alpha_scores_: dict[float, float] = {}

    def _build(self, alpha: float, task: str):
        from sklearn.linear_model import LogisticRegression, Ridge

        if task == CLASSIFICATION:
            # sklearn parameterises the logistic penalty as C = 1 / alpha, so a
            # large alpha here means the same thing it means for Ridge: more
            # shrinkage, not less.
            return LogisticRegression(
                penalty="l2",
                C=1.0 / alpha,
                solver="lbfgs",
                max_iter=self.max_iter,
                fit_intercept=self.fit_intercept,
            )
        return Ridge(alpha=alpha, fit_intercept=self.fit_intercept)

    def _validation_loss(self, estimator, X_val: pd.DataFrame, y_val: pd.Series, task: str) -> float:
        """Loss on the held-out tail of the training window. Lower is better."""
        from sklearn.metrics import log_loss, mean_squared_error

        if task == CLASSIFICATION:
            proba = estimator.predict_proba(X_val)[:, 1]
            return float(log_loss(y_val.astype(int), proba, labels=[0, 1]))
        return float(mean_squared_error(y_val.astype(float), estimator.predict(X_val)))

    def fit(self, fold: FoldData) -> "RidgeModel":
        self.feature_names_ = list(fold.X_train.columns)
        task = fold.task

        if task == CLASSIFICATION:
            y_train = fold.y_train.astype(int)
            y_val = fold.y_val.astype(int)
        else:
            y_train = fold.y_train.astype(float)
            y_val = fold.y_val.astype(float)

        # Same guard the boosted model uses for early stopping. The splitter can
        # hand back a val slice that is empty or single class near the start of
        # the walk forward, and a grid scored on that is worse than no grid.
        has_validation = len(fold.X_val) > 0 and y_val.nunique() > 1
        candidates = self.alphas if has_validation else [self.default_alpha]

        best_estimator, best_loss, best_alpha = None, np.inf, self.default_alpha
        for alpha in candidates:
            estimator = self._build(alpha, task)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                estimator.fit(fold.X_train, y_train)

            if not has_validation:
                best_estimator, best_alpha = estimator, alpha
                break

            loss = self._validation_loss(estimator, fold.X_val, y_val, task)
            self.alpha_scores_[alpha] = loss
            if loss < best_loss:
                best_estimator, best_loss, best_alpha = estimator, loss, alpha

        # The winning fit stays as it was trained, on the training rows only.
        # Refitting on train plus validation would spend the slice that chose
        # the hyper-parameter, which is the same mistake early stopping avoids.
        self.estimator_ = best_estimator
        self.best_alpha_ = best_alpha
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        if self.estimator_ is None:
            return np.zeros(len(X))
        return np.asarray(self.estimator_.predict(X[self.feature_names_]), dtype=float)

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray | None:
        if self.estimator_ is None or self.task != CLASSIFICATION:
            return None
        return np.asarray(
            self.estimator_.predict_proba(X[self.feature_names_])[:, 1], dtype=float
        )

    def feature_importance(self) -> pd.Series | None:
        """Absolute coefficient per feature.

        Magnitudes are comparable across columns only because the matrix was
        standardised on the training fold, so this is a like-for-like ranking
        rather than a reflection of which feature happened to be measured in
        larger units. Use :meth:`coefficients` when the sign matters.

        The runner pools this into ``feature_importance.csv`` under a column
        named ``gain``, which is the boosted model's vocabulary rather than
        this one's. The ``model`` column is what tells the two apart.
        """
        coefficients = self.coefficients()
        if coefficients is None:
            return None
        return coefficients.abs().sort_values(ascending=False)

    def coefficients(self) -> pd.Series | None:
        """Signed weights, in feature order. None before the model is fitted."""
        if self.estimator_ is None:
            return None
        coef = np.asarray(self.estimator_.coef_, dtype=float).ravel()
        return pd.Series(coef, index=self.feature_names_)
