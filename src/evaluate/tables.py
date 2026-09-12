"""Scoring and comparison tables built from a run's predictions.

Everything here takes predictions and truth and returns a table. Nothing here
fits a model or reads a config, which is what lets the same functions score a
single fold inside the runner, an asset within a finished run, and a run
directory reopened weeks later.

Two rules hold throughout. A stochastic model is averaged over its seeds
before any comparison, because you would not get to pick the lucky seed in
advance. And every test runs under the loss its headline metric is computed on,
because two columns that disagree about the same forecasts read as a finding
when they are a choice of loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema
from src.evaluate import metrics as metric_lib
from src.evaluate.diebold_mariano import diebold_mariano
from src.models.base import Prediction


def seed_averaged(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """One forecast per (asset, date) and model, averaged over seeds.

    Rows where ``column`` is missing are dropped first, so a model that made no
    forecast for a row contributes nothing rather than a NaN that would poison
    the mean. Returns a frame indexed by (asset, date) with one column per
    model; a model with no forecasts at all has no column.
    """
    usable = predictions.dropna(subset=[column])
    if usable.empty:
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=[schema.ASSET, schema.DATE]))
    return (
        usable.groupby([schema.ASSET, schema.DATE, "model"], observed=True)[column]
        .mean()
        .unstack("model")
    )


def score_prediction(
    y_true: pd.Series,
    prediction: Prediction,
    task: str,
    target: str,
) -> dict[str, float]:
    """Score a prediction, routing the volatility target to its own loss functions.

    ``vol_1d`` is registered as a regression target and stored on a log scale,
    but the volatility forecasting literature scores on QLIKE and the
    Mincer-Zarnowitz regression, both of which are defined on variances. RMSE
    on log variance is not wrong, it just answers a different question and
    penalises over and under prediction symmetrically in a place where that is
    not what a user of the forecast cares about. Converting at the boundary and
    dispatching on the volatility task is what makes the reported numbers the
    ones the target actually calls for.

    This is the one scoring path, used for a fold, for an asset within a run,
    and for the pooled row those asset numbers must add back up to.
    """
    if target == "vol_1d":
        variance_true = pd.Series(np.exp(y_true.to_numpy()), index=y_true.index)
        variance_pred = Prediction(
            index=prediction.index,
            point=np.exp(prediction.point),
            model_name=prediction.model_name,
            fold_id=prediction.fold_id,
            seed=prediction.seed,
        )
        row = metric_lib.evaluate_predictions(
            variance_true, variance_pred, metric_lib.VOLATILITY
        )
        # Keep the log scale errors alongside, since they are what the models
        # were fitted to and they make the folds comparable across assets.
        log_row = metric_lib.evaluate_predictions(y_true, prediction, task)
        row["rmse_log"] = log_row.get("rmse", float("nan"))
        row["mae_log"] = log_row.get("mae", float("nan"))
        return row

    return metric_lib.evaluate_predictions(y_true, prediction, task)


def default_dm_loss(target: str) -> str:
    """The loss the Diebold-Mariano test is run under for a given target.

    It has to be the loss the headline metric is computed on, or the two
    columns can disagree about the same forecasts. Direction is scored on
    Brier, which is squared loss on the probability. Volatility is scored on
    QLIKE, which is asymmetric: under-predicting variance costs far more than
    over-predicting it by the same factor, and squared error on log variance
    does not know that. A GARCH that never under-predicts can win on QLIKE and
    lose on log squared error at the same time, which is not a contradiction,
    but a table that shows both without saying so reads as one.
    """
    return "qlike" if target == "vol_1d" else "squared"


def default_dm_baseline(target: str) -> str:
    """The model a Diebold-Mariano test is run against for a given target.

    Direction is compared against the abstaining forecast, because a constant
    0.5 is the thing a probabilistic model actually has to beat. Volatility is
    compared against the trailing mean, which is the baseline that wins there.
    """
    return "vol_climatology" if target == "vol_1d" else "zero"


def dm_table(
    predictions: pd.DataFrame,
    y: pd.Series,
    baseline: str = "persistence",
    loss: str = "squared",
) -> pd.DataFrame:
    """Diebold-Mariano test of every model against one baseline.

    A negative statistic with a small p-value means the model genuinely has the
    lower loss. This is the column that separates "looks better" from "is
    better", and it is the one almost no student project reports.
    """
    if predictions.empty:
        return pd.DataFrame()

    # For a classification target the comparison has to run on probabilities,
    # not on hard labels. Squared loss against a 0/1 label scores 0 or 1 while
    # a constant 0.5 forecast always scores 0.25, so comparing the two measures
    # the output encoding rather than the quality of the forecast, and it makes
    # the abstaining baseline look unbeatable.
    column = "proba" if "proba" in predictions.columns else "pred"
    pooled = seed_averaged(predictions, column)
    if baseline not in pooled.columns:
        return pd.DataFrame()

    truth = y.reindex(pooled.index)
    if loss == "qlike":
        # QLIKE is defined on variances. The vol_1d target and its forecasts
        # are stored as log variance, so both sides are exponentiated here,
        # exactly as score_prediction does before computing the headline QLIKE.
        pooled = np.exp(pooled)
        truth = np.exp(truth)
    rows = []
    for model in pooled.columns:
        if model == baseline:
            continue
        mask = truth.notna() & pooled[model].notna() & pooled[baseline].notna()
        result = diebold_mariano(
            truth[mask].to_numpy(),
            pooled.loc[mask, model].to_numpy(),
            pooled.loc[mask, baseline].to_numpy(),
            loss=loss,
        )
        rows.append(
            {
                "model": model,
                "vs_baseline": baseline,
                "loss": loss,
                "dm_statistic": result.statistic,
                "p_value": result.p_value,
                "mean_loss_diff": result.mean_loss_diff,
                "better": result.better,
                "n_obs": result.n_obs,
                "note": result.note,
            }
        )
    return pd.DataFrame(rows)


def dm_per_asset(
    predictions: pd.DataFrame,
    y: pd.Series,
    baseline: str = "persistence",
    loss: str = "squared",
) -> pd.DataFrame:
    """The Diebold-Mariano test run separately inside each asset's rows.

    A model can lose to the baseline pooled and still beat it on one asset, or
    the reverse. This is the table that says which, with the sample size of
    each test next to its p-value so the reader can see how much it can detect.
    """
    if predictions.empty:
        return pd.DataFrame()

    assets = predictions.index.get_level_values(schema.ASSET)
    tables = []
    for asset in sorted(assets.unique()):
        subset = predictions[assets == asset]
        table = dm_table(subset, y, baseline=baseline, loss=loss)
        if not table.empty:
            table.insert(1, "asset", str(asset))
            tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def per_asset_results(
    predictions: pd.DataFrame, y: pd.Series, task: str, target: str
) -> pd.DataFrame:
    """Break the same predictions down by asset, with a pooled ``all`` row.

    Skill may exist on one asset and nowhere else, which would be a real and
    explicable finding rather than noise. Pooling across assets would hide it.

    Rows are scored pooled across folds and, for a stochastic model, across
    seeds, so every seed's forecast counts as a row. The ``all`` row scores the
    same rows without the asset split, so the asset numbers, weighted by their
    row counts, average back to it exactly and a reader can reconcile the
    breakdown against the headline.
    """
    if predictions.empty:
        return pd.DataFrame()

    assets = predictions.index.get_level_values(schema.ASSET)
    rows = []
    for (model, asset), group in predictions.groupby(["model", assets], observed=True):
        rows.append(_score_group(group, y, task, target, model=str(model), asset=str(asset)))
    for model, group in predictions.groupby("model", observed=True):
        rows.append(_score_group(group, y, task, target, model=str(model), asset="all"))
    table = pd.DataFrame(rows)
    front = ["model", "asset", "n_folds"]
    return table[front + [c for c in table.columns if c not in front]]


def _score_group(
    group: pd.DataFrame, y: pd.Series, task: str, target: str, model: str, asset: str
) -> dict:
    truth = y.reindex(group.index)
    proba = group["proba"].to_numpy() if "proba" in group else None
    prediction = Prediction(
        index=group.index,
        point=group["pred"].to_numpy(),
        proba=proba,
        model_name=model,
    )
    row = score_prediction(truth, prediction, task, target)
    row.update(
        {"model": model, "asset": asset, "n_folds": int(group["fold"].nunique()) if "fold" in group else 0}
    )
    return row
