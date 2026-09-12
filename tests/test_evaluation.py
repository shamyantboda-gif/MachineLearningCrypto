"""Contracts for the model registry and the Diebold-Mariano wiring.

Two gaps motivate this file. ``build_models`` had no test at all, and the fold
loop in ``src.train`` catches every model exception and only prints it, so a
mistyped family name or a broken constructor produces an empty results table
and no traceback. And ``dm_table`` was defined but never called from anywhere
in the pipeline, which meant the test the README leans on hardest had no
executable path behind it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.models.base import CLASSIFICATION, REGRESSION
from src.evaluate.report import scope_comparison
from src.evaluate.tables import (
    default_dm_baseline,
    default_dm_loss,
    dm_per_asset,
    dm_table,
    per_asset_results,
)
from src.train import build_models

# Family name -> the config that selects it and the class it must produce.
REGISTRY_CASES = {
    "ridge": ("config/models/ridge.yaml", "RidgeModel"),
    "lightgbm": ("config/models/lightgbm.yaml", "LightGBMModel"),
    "arima": ("config/models/arima.yaml", "ArimaModel"),
}


@pytest.mark.parametrize("family", sorted(REGISTRY_CASES))
def test_build_models_returns_the_family_its_config_names(family):
    """A model config resolves to its own class, not a neighbour's."""
    path, expected = REGISTRY_CASES[family]
    config = load_config(path)
    assert config["model"]["family"] == family

    models = build_models(config, task=CLASSIFICATION, seed=42)

    assert models, f"{family!r} built no models"
    assert type(models[0]).__name__ == expected
    assert models[0].name == family


def test_build_models_rejects_an_unknown_family():
    with pytest.raises(ValueError, match="unknown model family"):
        build_models({"model": {"family": "not_a_model"}}, task=CLASSIFICATION, seed=42)


def test_baselines_config_defers_to_the_baseline_builder():
    """Baselines are added to every run separately, so the family builds none."""
    assert build_models(load_config("config/models/baselines.yaml"), CLASSIFICATION, 42) == []


@pytest.fixture
def predictions_and_truth():
    """A run-shaped predictions frame: two models over a (asset, date) index.

    ``sharp`` gets the labels right more often than ``vague``, so the test has a
    known sign to assert on rather than checking that the call merely returns.
    The probabilities carry noise on purpose: a model whose loss differential is
    constant has zero variance, and the DM statistic is then genuinely
    undefined rather than merely inconvenient.
    """
    rng = np.random.default_rng(0)
    index = pd.MultiIndex.from_product(
        [["BTC", "ETH"], pd.bdate_range("2021-01-01", periods=250)],
        names=["asset", "date"],
    )
    truth = pd.Series(rng.integers(0, 2, len(index)), index=index, dtype=float)

    frames = []
    for model, weight in [("sharp", 0.75), ("vague", 0.5)]:
        centre = np.where(truth.to_numpy() == 1, weight, 1.0 - weight)
        proba = np.clip(centre + rng.normal(scale=0.05, size=len(index)), 0.01, 0.99)
        frames.append(
            pd.DataFrame(
                {"pred": (proba > 0.5).astype(float), "proba": proba, "model": model},
                index=index,
            )
        )
    return pd.concat(frames), truth


def test_dm_table_scores_every_model_against_the_baseline(predictions_and_truth):
    predictions, truth = predictions_and_truth

    table = dm_table(predictions, truth, baseline="vague")

    assert list(table["model"]) == ["sharp"], "the baseline must not be scored against itself"
    row = table.iloc[0]
    assert row["vs_baseline"] == "vague"
    assert row["n_obs"] == len(truth)
    # sharp carries real information, so its Brier loss is lower and the
    # statistic is negative by the convention in diebold_mariano.
    assert row["mean_loss_diff"] < 0
    assert row["dm_statistic"] < 0
    assert row["better"] == "a"


def test_dm_table_is_empty_when_the_baseline_is_absent(predictions_and_truth):
    """A vol run asking for `zero` degrades to no table rather than an error."""
    predictions, truth = predictions_and_truth
    assert dm_table(predictions, truth, baseline="not_a_model").empty


def test_dm_table_survives_an_empty_run():
    assert dm_table(pd.DataFrame(), pd.Series(dtype=float), baseline="zero").empty


@pytest.mark.parametrize(
    ("target", "expected"),
    [("dir_1d", "zero"), ("ret_1d", "zero"), ("vol_1d", "vol_climatology")],
)
def test_default_dm_baseline_matches_the_baselines_each_target_builds(target, expected):
    """The default must name a model that target's run actually produces."""
    from src.models.baselines import build_baselines

    assert default_dm_baseline(target) == expected

    task = CLASSIFICATION if target == "dir_1d" else REGRESSION
    built = {model.name for model in build_baselines(target, params={}, task=task)}
    assert expected in built, f"{expected!r} is not among {sorted(built)} for {target}"


def test_per_asset_results_scores_each_asset_and_reconciles_to_the_pooled_row(
    predictions_and_truth,
):
    """One row per model and asset, plus an ``all`` row on the same rows pooled.

    The ``all`` row is what lets a reader check the per-asset numbers against
    the headline: weighted by row count, the asset accuracies must average to
    it exactly.
    """
    predictions, truth = predictions_and_truth

    table = per_asset_results(predictions, truth, task=CLASSIFICATION, target="dir_1d")

    assert set(table["asset"]) == {"BTC", "ETH", "all"}
    assert set(table["model"]) == {"sharp", "vague"}
    assert len(table) == 6

    sharp = table[table["model"] == "sharp"].set_index("asset")
    by_asset = sharp.drop(index="all")
    weighted = (by_asset["directional_accuracy"] * by_asset["n_obs"]).sum() / by_asset["n_obs"].sum()
    assert weighted == pytest.approx(sharp.loc["all", "directional_accuracy"])
    assert sharp.loc["all", "n_obs"] == len(truth)
    assert sharp.loc["BTC", "directional_accuracy"] > sharp.loc["BTC", "base_rate"]
    assert "roc_auc" in table.columns and "brier_score" in table.columns


def test_dm_per_asset_runs_the_test_inside_each_asset(predictions_and_truth):
    predictions, truth = predictions_and_truth

    table = dm_per_asset(predictions, truth, baseline="vague")

    assert set(table["asset"]) == {"BTC", "ETH"}
    assert list(table["model"].unique()) == ["sharp"]
    assert (table["n_obs"] == 250).all(), "each asset is tested on its own 250 rows"
    assert (table["dm_statistic"] < 0).all()


def test_scope_comparison_pairs_pooled_and_per_asset_fits_on_common_rows(
    predictions_and_truth,
):
    """``ridge`` and ``ridge_per_asset`` are compared only where both predicted.

    The per-asset arm drops the first 50 rows of ETH, as it would when an asset
    has too little history to fit on. The comparison must shrink to the rows
    both arms cover rather than score the two on different samples.
    """
    predictions, truth = predictions_and_truth
    pooled = predictions[predictions["model"] == "sharp"].assign(model="ridge")
    per_asset = predictions[predictions["model"] == "vague"].assign(model="ridge_per_asset")
    eth_rows = per_asset.index.get_level_values("asset") == "ETH"
    drop = per_asset.index[eth_rows][:50]
    per_asset = per_asset.drop(index=drop)
    frame = pd.concat([pooled, per_asset])
    frame["y_true"] = truth.reindex(frame.index)

    table = scope_comparison(frame)

    assert set(table["family"]) == {"ridge"}
    assert set(table["asset"]) == {"BTC", "ETH", "all"}
    by_asset = table.set_index("asset")
    assert by_asset.loc["BTC", "n_obs"] == 250
    assert by_asset.loc["ETH", "n_obs"] == 200
    assert by_asset.loc["all", "n_obs"] == 450
    # sharp (pooled) is the better forecast, so per-asset loses at every level.
    assert (table["accuracy_pooled"] > table["accuracy_per_asset"]).all()
    assert (table["dm_statistic"] > 0).all()
    assert (table["p_value"] < 0.05).all()


def test_scope_comparison_is_empty_without_a_per_asset_arm(predictions_and_truth):
    predictions, truth = predictions_and_truth
    frame = predictions.assign(y_true=truth.reindex(predictions.index))
    assert scope_comparison(frame).empty


@pytest.mark.parametrize(
    ("target", "expected"),
    [("dir_1d", "squared"), ("ret_1d", "squared"), ("vol_1d", "qlike")],
)
def test_default_dm_loss_matches_the_headline_metric(target, expected):
    assert default_dm_loss(target) == expected


def test_dm_table_under_qlike_scores_variances_not_log_variances():
    """A log-variance run tested under QLIKE must exponentiate first.

    Two forecasts of log variance: ``tight`` is the truth plus small noise,
    ``under`` is the truth minus one, so it under-predicts variance by a factor
    of e everywhere. Under squared loss on the log scale, ``under`` is off by a
    constant 1.0 and ``tight`` by ~0.01, so both losses agree that ``tight`` is
    better. The point of the test is that the QLIKE path runs at all on this
    input, which it cannot if the frame is passed through as logs: QLIKE
    raises on a non-positive "variance", and log variances are negative.
    """
    rng = np.random.default_rng(3)
    index = pd.MultiIndex.from_product(
        [["BTC"], pd.date_range("2021-01-01", periods=300, freq="D", tz="UTC")],
        names=["asset", "date"],
    )
    truth = pd.Series(rng.normal(loc=-7.0, scale=0.5, size=len(index)), index=index)
    frames = [
        pd.DataFrame({"pred": truth + rng.normal(scale=0.1, size=len(index)), "model": "tight"}, index=index),
        pd.DataFrame({"pred": truth - 1.0, "model": "under"}, index=index),
    ]
    predictions = pd.concat(frames)

    table = dm_table(predictions, truth, baseline="under", loss="qlike")

    assert list(table["model"]) == ["tight"]
    row = table.iloc[0]
    assert row["loss"] == "qlike"
    assert row["dm_statistic"] < 0 and row["better"] == "a"
    assert np.isfinite(row["p_value"])


def test_base_rate_is_computed_on_the_rows_the_model_scored():
    """A skipped asset's up-days must not enter the base rate its model is read against."""
    from src.evaluate import metrics
    from src.models.base import Prediction

    index = pd.MultiIndex.from_product(
        [["BTC", "ETH"], pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC")],
        names=["asset", "date"],
    )
    # BTC: one up-day in four. ETH: all up-days, and no forecasts.
    truth = pd.Series([1, 0, 0, 0, 1, 1, 1, 1], index=index, dtype=float)
    proba = np.array([0.6, 0.4, 0.4, 0.6, np.nan, np.nan, np.nan, np.nan])
    row = metrics.evaluate_predictions(
        truth, Prediction(index=index, point=(proba >= 0.5).astype(float), proba=proba), CLASSIFICATION
    )
    assert row["n_obs"] == 4
    assert row["base_rate"] == pytest.approx(0.25), "base rate leaked ETH's unscored up-days"


def test_scope_comparison_skips_an_arm_that_never_predicted(predictions_and_truth):
    predictions, truth = predictions_and_truth
    pooled = predictions[predictions["model"] == "sharp"].assign(model="ridge")
    empty_arm = pooled.assign(model="ridge_per_asset", pred=np.nan, proba=np.nan)
    frame = pd.concat([pooled, empty_arm])
    frame["y_true"] = truth.reindex(frame.index)
    assert scope_comparison(frame).empty
