"""LightGBM on the tabular feature matrix.

This is the strong classical benchmark, not a warm-up act. On tabular financial
features gradient boosting frequently beats deep sequence models, trains in
seconds rather than minutes, and hands back SHAP attributions for free.

The configuration is deliberately conservative: shallow trees, a high minimum
leaf population and heavy L2. In a domain where the signal to noise ratio is
close to zero, a deep tree does not find structure, it memorises which day was
which. The regularisation is doing most of the work here and the defaults in
``config/models/lightgbm.yaml`` reflect that.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.models.base import CLASSIFICATION, FoldData, Model


class LightGBMModel(Model):
    """Gradient boosted trees with early stopping on the inner validation slice."""

    name = "lightgbm"
    is_stochastic = True

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.model_params = dict(self.params.get("params", {}))
        self.early_stopping_rounds = int(self.params.get("early_stopping_rounds", 100))
        self.want_shap = bool(self.params.get("shap", False))

        self.booster_ = None
        self.feature_names_: list[str] = []
        self.best_iteration_: int | None = None

    def fit(self, fold: FoldData) -> "LightGBMModel":
        import lightgbm as lgb

        self.feature_names_ = list(fold.X_train.columns)

        params = dict(self.model_params)
        params.setdefault("verbosity", -1)
        params["random_state"] = self.seed
        params["n_jobs"] = -1

        if fold.task == CLASSIFICATION:
            estimator = lgb.LGBMClassifier(objective="binary", **params)
            y_train = fold.y_train.astype(int)
            y_val = fold.y_val.astype(int)
        else:
            estimator = lgb.LGBMRegressor(objective="regression", **params)
            y_train = fold.y_train.astype(float)
            y_val = fold.y_val.astype(float)

        callbacks = []
        has_validation = len(fold.X_val) > 0 and y_val.nunique() > 1
        if has_validation and self.early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            )
        callbacks.append(lgb.log_evaluation(period=0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            estimator.fit(
                fold.X_train,
                y_train,
                eval_set=[(fold.X_val, y_val)] if has_validation else None,
                callbacks=callbacks,
            )

        self.booster_ = estimator
        self.best_iteration_ = getattr(estimator, "best_iteration_", None)
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        if self.booster_ is None:
            return np.zeros(len(X))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(self.booster_.predict(X[self.feature_names_]), dtype=float)

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray | None:
        if self.booster_ is None or self.task != CLASSIFICATION:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                self.booster_.predict_proba(X[self.feature_names_])[:, 1], dtype=float
            )

    def feature_importance(self) -> pd.Series | None:
        """Gain-based importance. Split counts flatter noise, gain does not."""
        if self.booster_ is None:
            return None
        booster = self.booster_.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=self.feature_names_).sort_values(ascending=False)

    def shap_values(self, X: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame | None:
        """Mean absolute SHAP value per feature on a sample of ``X``.

        Instability of the importance ranking across folds is itself a finding
        worth reporting, so this is computed per fold rather than once.
        """
        if self.booster_ is None or not self.want_shap:
            return None
        try:
            import shap
        except ImportError:
            return None

        sample = X[self.feature_names_]
        if len(sample) > max_rows:
            sample = sample.sample(max_rows, random_state=self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(self.booster_.booster_)
            values = explainer.shap_values(sample)

        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]

        return pd.DataFrame(
            {
                "feature": self.feature_names_,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)
