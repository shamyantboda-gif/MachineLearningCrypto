"""Contracts for fitting one model per asset instead of one across the panel.

The wrapper is the only thing between "per asset" in a config file and
"per asset" in a results table, so it has to be checked for the two ways it
could silently be wrong: rows of one asset reaching another asset's model, and
a skipped asset producing a number instead of a gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.base import CLASSIFICATION, FoldData
from src.models.per_asset import PerAssetModel
from src.models.ridge import RidgeModel
from src.train import build_models, load_model_config


def _fold(n_train: int = 400, n_test: int = 60, flip_asset: str = "ETH") -> FoldData:
    """Two assets whose labels point in opposite directions.

    ``dir`` is ``x > 0`` for BTC and ``x < 0`` for ETH. One linear model over
    both assets can only learn one of the two rules; one model per asset gets
    both. That difference is the whole reason the wrapper exists, so it is the
    thing the tests measure.
    """
    rng = np.random.default_rng(0)
    frames, labels = [], []
    dates = pd.date_range("2021-01-01", periods=n_train + n_test, freq="D", tz="UTC")
    for asset in ["BTC", "ETH"]:
        x = rng.normal(size=len(dates))
        noise = rng.normal(scale=0.1, size=len(dates))
        index = pd.MultiIndex.from_product([[asset], dates], names=["asset", "date"])
        frames.append(pd.DataFrame({"x": x, "z": noise}, index=index))
        sign = -1.0 if asset == flip_asset else 1.0
        labels.append(pd.Series((sign * x > 0).astype(float), index=index))
    X = pd.concat(frames).sort_index()
    y = pd.concat(labels).sort_index()
    is_test = X.index.get_level_values("date") >= dates[n_train]
    meta = pd.DataFrame({"close": 1.0}, index=X.index)
    return FoldData(
        fold_id=0,
        X_train=X[~is_test],
        y_train=y[~is_test],
        X_val=X[~is_test].iloc[:0],
        y_val=y[~is_test].iloc[:0],
        X_test=X[is_test],
        y_test=y[is_test],
        meta_train=meta[~is_test],
        meta_test=meta[is_test],
        train_start=dates[0],
        train_end=dates[n_train - 1],
        test_start=dates[n_train],
        test_end=dates[-1],
        target="dir_1d",
    )


def _ridge(seed: int = 0) -> RidgeModel:
    return RidgeModel(params={"alphas": [1.0], "default_alpha": 1.0}, task=CLASSIFICATION, seed=seed)


def test_per_asset_fit_learns_a_rule_the_pooled_fit_cannot():
    fold = _fold()
    pooled = _ridge().run_fold(fold)
    per_asset = PerAssetModel(_ridge, min_train_rows=10, task=CLASSIFICATION).run_fold(fold)

    truth = fold.y_test.to_numpy()
    pooled_accuracy = np.mean((pooled.proba >= 0.5) == (truth == 1))
    per_asset_accuracy = np.mean((per_asset.proba >= 0.5) == (truth == 1))
    assert pooled_accuracy < 0.7, "one linear rule cannot fit two opposite signs"
    assert per_asset_accuracy > 0.95


def test_per_asset_predictions_align_to_the_test_index_and_name_the_scope():
    fold = _fold()
    model = PerAssetModel(_ridge, min_train_rows=10, task=CLASSIFICATION)
    prediction = model.run_fold(fold)

    assert model.name == "ridge_per_asset"
    assert prediction.index.equals(fold.X_test.index)
    assert len(prediction.point) == len(fold.X_test)
    assert np.isfinite(prediction.proba).all()
    assert set(model.models_) == {"BTC", "ETH"}
    # Two genuinely different fits, not one model stored twice.
    a = model.models_["BTC"].estimator_.coef_[0, 0]
    b = model.models_["ETH"].estimator_.coef_[0, 0]
    assert np.sign(a) != np.sign(b)


def test_an_asset_below_the_row_floor_is_skipped_not_fitted_on_scraps():
    fold = _fold()
    # Cut ETH's training history to 30 rows; BTC keeps all 400.
    eth_train = fold.X_train.index.get_level_values("asset") == "ETH"
    keep = ~eth_train | (fold.X_train.index.get_level_values("date") >= fold.train_end - pd.Timedelta(days=29))
    fold.X_train = fold.X_train[keep]
    fold.y_train = fold.y_train[keep]
    fold.meta_train = fold.meta_train[keep]

    model = PerAssetModel(_ridge, min_train_rows=100, task=CLASSIFICATION)
    prediction = model.run_fold(fold)

    assets = prediction.index.get_level_values("asset")
    assert set(model.models_) == {"BTC"}
    assert model.skipped_ == {"ETH": 30}
    assert np.isnan(prediction.proba[assets == "ETH"]).all()
    assert np.isnan(prediction.point[assets == "ETH"]).all()
    assert np.isfinite(prediction.proba[assets == "BTC"]).all()


def test_per_asset_feature_importance_averages_the_asset_models():
    fold = _fold()
    model = PerAssetModel(_ridge, min_train_rows=10, task=CLASSIFICATION).fit(fold)
    importance = model.feature_importance()
    assert importance is not None
    assert list(importance.index) == ["x", "z"]
    assert importance["x"] > importance["z"]


@pytest.mark.parametrize("family", ["ridge", "lightgbm"])
def test_per_asset_configs_extend_the_pooled_config_and_wrap_the_family(family):
    """The per-asset YAML inherits the pooled family's parameters.

    The two arms must differ in scope and nothing else, or a gap between them
    could be a hyper-parameter rather than the thing being tested.
    """
    pooled = load_model_config(f"config/models/{family}.yaml")
    per_asset = load_model_config(f"config/models/{family}_per_asset.yaml")

    assert per_asset["model"]["fit_scope"] == "per_asset"
    assert per_asset["model"]["min_train_rows"] == load_config("config/base.yaml")["splits"]["min_train_rows"]
    resolved = {k: v for k, v in per_asset["model"].items() if k not in {"fit_scope", "min_train_rows", "extends"}}
    assert resolved == pooled["model"]

    built = build_models(per_asset, CLASSIFICATION, seed=42)
    assert len(built) == 1
    assert isinstance(built[0], PerAssetModel)
    assert built[0].name == f"{family}_per_asset"
    assert built[0].is_stochastic == build_models(pooled, CLASSIFICATION, seed=42)[0].is_stochastic
