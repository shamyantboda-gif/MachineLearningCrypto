"""Fit one copy of a model family per asset instead of one across the panel.

The pooled fit sees every asset's rows at once and learns whatever structure
they share, which for crypto is most of it, on roughly three times the sample.
The per-asset fit gives up that sample in exchange for parameters that are
allowed to differ by asset. Which trade wins is an empirical question, and the
way to answer it is to run both on the same folds and rows and compare, so this
wrapper turns any family into its per-asset version without the family knowing.

The wrapper slices the fold by asset, fits a fresh inner model on each slice,
and reassembles the predictions onto the fold's own test index. An asset with
fewer training rows than ``min_train_rows`` is skipped for that fold and its
test rows come back as NaN, which every consumer downstream already treats as
"no forecast": the scorer drops them, the Diebold-Mariano test drops them, and
the backtest holds no position. The gap is recorded in ``skipped_`` so a report
can say how many rows the per-asset arm never predicted, rather than a smaller
sample quietly passing as the same one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

import numpy as np
import pandas as pd

from src import schema
from src.models.base import CLASSIFICATION, FoldData, Model

SCOPE_SUFFIX = "_per_asset"


class PerAssetModel(Model):
    """One inner model per asset, behind the ordinary ``Model`` interface."""

    def __init__(
        self,
        build: Callable[[], Model],
        min_train_rows: int = 500,
        task: str = CLASSIFICATION,
        seed: int = 42,
    ):
        super().__init__(params={"min_train_rows": int(min_train_rows)}, task=task, seed=seed)
        self._build = build
        template = build()
        self.name = f"{template.name}{SCOPE_SUFFIX}"
        self.is_stochastic = template.is_stochastic
        self.min_train_rows = int(min_train_rows)
        self.models_: dict[str, Model] = {}
        self.skipped_: dict[str, int] = {}

    def fit(self, fold: FoldData) -> "PerAssetModel":
        self.models_, self.skipped_ = {}, {}
        for asset in _assets(fold.X_train):
            sub = _slice_fold(fold, asset)
            if len(sub.X_train) < self.min_train_rows:
                self.skipped_[asset] = len(sub.X_train)
                continue
            model = self._build()
            model.fit(sub)
            self.models_[asset] = model
        self.fitted_ = True
        return self

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        return self._assemble(X, meta, lambda model, x, m: model.predict(x, m))

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray | None:
        if self.task != CLASSIFICATION:
            return None
        return self._assemble(X, meta, lambda model, x, m: model.predict_proba(x, m))

    def _assemble(self, X: pd.DataFrame, meta: pd.DataFrame | None, call) -> np.ndarray:
        """Run each asset's model on its own rows and write back by label.

        Writing by label rather than by concatenation is what keeps a
        reordered test frame, or one with an asset the wrapper never fitted,
        from producing values on the wrong rows.
        """
        out = pd.Series(np.nan, index=X.index, dtype=float)
        assets = X.index.get_level_values(schema.ASSET)
        for asset, model in self.models_.items():
            mask = assets == asset
            if not mask.any():
                continue
            rows = X[mask]
            sub_meta = meta[mask] if meta is not None else None
            values = call(model, rows, sub_meta)
            if values is None:
                continue
            out.loc[rows.index] = np.asarray(values, dtype=float)
        return out.to_numpy()

    def feature_importance(self) -> pd.Series | None:
        """Mean importance across the asset models that report one."""
        series = [m.feature_importance() for m in self.models_.values()]
        series = [s for s in series if s is not None]
        if not series:
            return None
        return pd.concat(series, axis=1).mean(axis=1)

    def __repr__(self) -> str:
        return f"PerAssetModel(name={self.name!r}, assets={sorted(self.models_)}, skipped={self.skipped_})"


def _assets(frame: pd.DataFrame) -> list[str]:
    return [str(a) for a in frame.index.get_level_values(schema.ASSET).unique()]


def _slice_fold(fold: FoldData, asset: str) -> FoldData:
    """The same fold restricted to one asset's rows, every split at once."""

    def rows(frame):
        return frame[frame.index.get_level_values(schema.ASSET) == asset]

    return replace(
        fold,
        X_train=rows(fold.X_train),
        y_train=rows(fold.y_train),
        X_val=rows(fold.X_val),
        y_val=rows(fold.y_val),
        X_test=rows(fold.X_test),
        y_test=rows(fold.y_test),
        meta_train=rows(fold.meta_train),
        meta_test=rows(fold.meta_test),
    )
