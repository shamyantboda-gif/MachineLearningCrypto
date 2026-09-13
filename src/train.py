"""Experiment runner.

    python -m src.train --config config/base.yaml --model config/models/lightgbm.yaml

Every run is fully specified by its merged config, and the hash of that config
names the output directory. Nothing about a run lives in a notebook or in a
shell history.

The loop is deliberately boring. For each walk-forward fold: fit a fresh
preprocessor on the training rows, transform all three splits with it, hand the
result to each model, collect predictions, score them against the same rows the
baselines were scored on. Stochastic models repeat that over several seeds,
because in this domain the seed to seed spread routinely exceeds the gap
between model families and a single-seed number is not a result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    RESULTS_DIR,
    config_hash,
    ensure_dirs,
    load_config,
    set_global_seed,
)
from src.data.build_panel import load_auxiliary, load_panel
from src.evaluate import metrics as metric_lib
from src.evaluate.tables import (
    default_dm_baseline,
    default_dm_loss,
    dm_per_asset,
    dm_table,
    per_asset_results,
    score_prediction,
)
from src.features.registry import build_features
from src.models.base import TARGET_TASKS, FoldData, Model, Prediction
from src.models.baselines import build_baselines
from src.preprocess import FoldPreprocessor
from src.splits.walk_forward import splitter_from_config


def load_model_config(path: str | Path) -> dict:
    """Load a model config, resolving ``model.extends`` against a sibling file.

    A per-asset config names the pooled config it inherits from and adds only
    ``fit_scope``, so the two arms cannot drift apart in a hyper-parameter. The
    resolved dictionary is what gets hashed, so the run directory still names
    the parameters that were actually used.
    """
    path = Path(path)
    config = load_config(path)
    parent = config.get("model", {}).get("extends")
    if parent is None:
        return config
    base = load_model_config(path.parent / parent)
    merged = dict(base["model"])
    merged.update({k: v for k, v in config["model"].items() if k != "extends"})
    return {**base, **config, "model": merged}


def build_models(model_config: dict, task: str, seed: int) -> list[Model]:
    """Instantiate the model family named by a model config file.

    ``fit_scope: per_asset`` wraps whatever the family would have been in a
    ``PerAssetModel`` that fits one copy per asset. The wrapper is built from a
    factory rather than an instance so a stochastic family can be rebuilt per
    seed the same way the pooled one is.
    """
    family = model_config["model"]["family"]
    params = model_config["model"]

    if family == "baselines":
        return []  # baselines are always added separately

    scope = params.get("fit_scope", "pooled")
    if scope == "per_asset":
        from src.models.per_asset import PerAssetModel

        inner = {
            **model_config,
            "model": {k: v for k, v in params.items() if k not in {"fit_scope", "min_train_rows"}},
        }
        pooled = build_models(inner, task, seed)
        return [
            PerAssetModel(
                build=lambda i=i: build_models(inner, task, seed)[i],
                min_train_rows=int(params.get("min_train_rows", 500)),
                task=task,
                seed=seed,
            )
            for i in range(len(pooled))
        ]
    if scope != "pooled":
        raise ValueError(f"unknown fit_scope {scope!r}, expected 'pooled' or 'per_asset'")

    if family == "arima":
        from src.models.arima import ArimaModel

        return [ArimaModel(params=params, task=task, seed=seed)]

    if family == "garch":
        from src.models.garch import build_garch_models

        return build_garch_models(model_config, task=task)

    if family == "ridge":
        from src.models.ridge import RidgeModel

        return [RidgeModel(params=params, task=task, seed=seed)]

    if family == "lightgbm":
        from src.models.gbm import LightGBMModel

        return [LightGBMModel(params=params, task=task, seed=seed)]

    if family in {"lstm", "cnn", "dlinear"}:
        from src.models.torch_common import build_torch_model

        return [build_torch_model(family, params, task, seed)]

    raise ValueError(f"unknown model family {family!r}")


def _fold_data(
    fold,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    config: dict,
    target: str,
) -> tuple[FoldData, FoldPreprocessor]:
    """Assemble one fold, fitting the preprocessor on training rows only."""
    preprocessor = FoldPreprocessor.from_config(config)
    X_train_raw = X.loc[fold.train_mask]

    preprocessor.fit(X_train_raw)

    data = FoldData(
        fold_id=fold.fold_id,
        X_train=preprocessor.transform(X_train_raw),
        y_train=y.loc[fold.train_mask],
        X_val=preprocessor.transform(X.loc[fold.val_mask]),
        y_val=y.loc[fold.val_mask],
        X_test=preprocessor.transform(X.loc[fold.test_mask]),
        y_test=y.loc[fold.test_mask],
        meta_train=meta.loc[fold.train_mask],
        meta_test=meta.loc[fold.test_mask],
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        target=target,
    )
    return data, preprocessor


def _score(
    data: FoldData,
    prediction: Prediction,
    task: str,
) -> dict[str, float]:
    """Score one fold on the rows the fold handed the model."""
    return score_prediction(data.y_test, prediction, task, data.target)


def run(
    config: dict,
    model_configs: list[dict],
    max_folds: int | None = None,
    quiet: bool = False,
    dm_baseline: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every configured model over every fold. Returns (results, predictions).

    ``dm_baseline`` deliberately stays out of ``config``. The run directory is
    named by a hash of the merged configuration, so a knob that only changes
    which comparison is reported must not change where the results land.
    """
    ensure_dirs()
    seed = int(config.get("seed", 42))
    set_global_seed(seed)

    target = config["targets"]["primary"]
    task = TARGET_TASKS[target]

    panel = load_panel()
    auxiliary = load_auxiliary()
    matrix = build_features(panel, config, auxiliary)

    X, y, meta = matrix.X, matrix.targets[target], matrix.meta
    splitter = splitter_from_config(config)
    folds = list(splitter.split(X.index))
    if max_folds is not None:
        folds = folds[:max_folds]

    if not quiet:
        print(f"target {target} ({task})")
        print(f"features {X.shape[1]} across {len(matrix.families)} families")
        print(f"rows {len(X):,}  folds {len(folds)}")
        print(matrix.summary().to_string(index=False))
        print()

    n_seeds = int(config.get("evaluate", {}).get("n_seeds", 5))

    result_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for fold in folds:
        data, _ = _fold_data(fold, X, y, meta, config, target)
        if not quiet:
            print(str(fold))

        fold_models: list[tuple[Model, int]] = []
        for baseline in build_baselines(target, params={}, task=task):
            fold_models.append((baseline, seed))
        for model_config in model_configs:
            for candidate in build_models(model_config, task, seed):
                if candidate.is_stochastic:
                    for offset in range(n_seeds):
                        fresh = build_models(model_config, task, seed + offset)
                        fold_models.extend((m, seed + offset) for m in fresh)
                    break
                fold_models.append((candidate, seed))

        for model, model_seed in fold_models:
            set_global_seed(model_seed)
            started = time.perf_counter()
            try:
                prediction = model.run_fold(data)
            except Exception as error:
                print(f"    {model.name} failed on fold {fold.fold_id}: {error}")
                continue
            elapsed = time.perf_counter() - started

            row = _score(data, prediction, task)
            row.update(
                {
                    "model": model.name,
                    "fold": fold.fold_id,
                    "seed": model_seed if model.is_stochastic else np.nan,
                    "target": target,
                    "test_start": fold.test_start.date(),
                    "test_end": fold.test_end.date(),
                    "n_train": fold.n_train,
                    "fit_seconds": round(elapsed, 3),
                }
            )
            result_rows.append(row)

            frame = prediction.to_frame()
            frame["target"] = target
            prediction_frames.append(frame)

            importance = model.feature_importance()
            if importance is not None:
                importance_frames.append(
                    importance.rename("gain")
                    .to_frame()
                    .assign(
                        model=model.name,
                        fold=fold.fold_id,
                        seed=model_seed if model.is_stochastic else np.nan,
                    )
                    .reset_index(names="feature")
                )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames) if prediction_frames else pd.DataFrame()
    importances = pd.concat(importance_frames) if importance_frames else pd.DataFrame()

    # The test that separates "looks better" from "is better". It is computed
    # here rather than left to the reader so that every p-value quoted about
    # this project has a file behind it.
    baseline = dm_baseline or default_dm_baseline(target)
    loss = default_dm_loss(target)
    dm = dm_table(predictions, y, baseline=baseline, loss=loss)
    if not quiet and not dm.empty:
        print(f"\nDiebold-Mariano vs {baseline}")
        print(dm[["model", "dm_statistic", "p_value", "better"]].to_string(index=False))

    # The same predictions, split by asset. A pooled number can hide skill
    # that lives on one asset, or hide one asset dragging the others down.
    per_asset = per_asset_results(predictions, y, task, target)
    per_asset_dm = dm_per_asset(predictions, y, baseline=baseline, loss=loss)

    # The truth travels with the predictions so a report can be rebuilt from
    # the run directory alone, without recomputing the feature matrix.
    if not predictions.empty:
        predictions["y_true"] = y.reindex(predictions.index).to_numpy()

    _persist(
        config,
        model_configs,
        results,
        predictions,
        importances,
        dm,
        per_asset,
        per_asset_dm,
        matrix,
        quiet,
    )
    return results, predictions


def _persist(
    config: dict,
    model_configs: list[dict],
    results: pd.DataFrame,
    predictions: pd.DataFrame,
    importances: pd.DataFrame,
    dm: pd.DataFrame,
    per_asset: pd.DataFrame,
    per_asset_dm: pd.DataFrame,
    matrix,
    quiet: bool,
) -> None:
    merged = {"base": config, "models": model_configs}
    run_id = config_hash(merged)
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(out_dir / "results.csv", index=False)
    if not predictions.empty:
        predictions.reset_index().to_parquet(out_dir / "predictions.parquet", index=False)
    if not importances.empty:
        importances.to_csv(out_dir / "feature_importance.csv", index=False)
    if not dm.empty:
        dm.to_csv(out_dir / "dm.csv", index=False)
    if not per_asset.empty:
        per_asset.to_csv(out_dir / "per_asset.csv", index=False)
    if not per_asset_dm.empty:
        per_asset_dm.to_csv(out_dir / "dm_per_asset.csv", index=False)

    (out_dir / "config.json").write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    (out_dir / "features.txt").write_text("\n".join(matrix.feature_names), encoding="utf-8")

    # A single flat index of every run, so the newest result is never the only
    # one you can find.
    index_path = RESULTS_DIR / "runs.csv"
    entry = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "target": config["targets"]["primary"],
                "models": ",".join(
                    sorted(
                        m["model"]["family"]
                        + ("_per_asset" if m["model"].get("fit_scope") == "per_asset" else "")
                        for m in model_configs
                    )
                ),
                "n_rows": len(results),
            }
        ]
    )
    if index_path.exists():
        entry = pd.concat([pd.read_csv(index_path), entry], ignore_index=True)
    entry.to_csv(index_path, index=False)

    if not quiet:
        print(f"\nwrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a walk-forward experiment.")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model config file, repeatable",
    )
    parser.add_argument("--target", default=None, help="override targets.primary")
    parser.add_argument("--scheme", default=None, help="override splits.scheme")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--dm-baseline",
        default=None,
        help="model the Diebold-Mariano test compares against, defaults per target",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target:
        config["targets"]["primary"] = args.target
    if args.scheme:
        config["splits"]["scheme"] = args.scheme

    model_configs = [load_model_config(path) for path in args.model]
    results, predictions = run(
        config,
        model_configs,
        max_folds=args.max_folds,
        dm_baseline=args.dm_baseline,
    )

    if results.empty:
        print("no results produced")
        return

    print("\nper-fold summary")
    print(metric_lib.summarise_folds(results).to_string())


if __name__ == "__main__":
    main()
