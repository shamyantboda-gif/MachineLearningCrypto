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
from src.train import build_models, default_dm_baseline, dm_table

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
