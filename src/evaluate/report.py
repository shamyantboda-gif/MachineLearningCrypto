"""Turn a run's raw outputs into the tables and figures a reader can act on.

Two reporting rules drive the whole module and both come from the project's
evaluation discipline:

Report per fold, never pooled. A model averaging 54 percent accuracy while
ranging from 44 to 63 across folds is a completely different object from one
sitting steadily at 54, and a single pooled number cannot tell them apart.

Report per asset and per regime. Skill may exist on the least efficient asset
in the panel and nowhere else, or only in one kind of market. Those are real
findings, and pooling destroys them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import schema
from src.config import FIGURES_DIR, RESULTS_DIR

# Metrics worth putting in a summary table, in the order a reader wants them.
CLASSIFICATION_COLUMNS = [
    "directional_accuracy",
    "base_rate",
    "balanced_accuracy",
    "matthews_corrcoef",
    "roc_auc",
    "brier_score",
]
REGRESSION_COLUMNS = ["rmse", "mae", "r2_oos"]


def latest_run(results_dir: Path = RESULTS_DIR) -> Path:
    """Most recently written run directory."""
    candidates = [p for p in results_dir.iterdir() if p.is_dir() and (p / "results.csv").exists()]
    if not candidates:
        raise FileNotFoundError(f"no completed runs under {results_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(run_dir / "results.csv")
    predictions_path = run_dir / "predictions.parquet"
    predictions = (
        schema.set_panel_index(pd.read_parquet(predictions_path))
        if predictions_path.exists()
        else pd.DataFrame()
    )
    return results, predictions


def fold_table(results: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Mean and spread across folds, one row per model.

    The standard deviation column is the one that matters. Where it is large
    relative to the gap between two models, those two models have not been
    distinguished by this experiment.
    """
    available = columns or [c for c in CLASSIFICATION_COLUMNS if c in results.columns]
    if not available:
        available = [c for c in REGRESSION_COLUMNS if c in results.columns]

    grouped = results.groupby("model")[available]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary["n_folds"] = results.groupby("model")["fold"].nunique()
    return summary.sort_values(summary.columns[0], ascending=False).round(4)


def edge_over_baseline(
    results: pd.DataFrame,
    metric: str = "directional_accuracy",
    baseline: str = "persistence",
) -> pd.DataFrame:
    """Per-fold difference between each model and one baseline on the same fold.

    Differencing within a fold rather than comparing two averages is what
    removes the fold's own difficulty from the comparison. A fold where every
    model scored badly says nothing about any of them.
    """
    if metric not in results.columns:
        return pd.DataFrame()

    wide = results.pivot_table(index="fold", columns="model", values=metric, aggfunc="mean")
    if baseline not in wide.columns:
        return pd.DataFrame()

    edge = wide.sub(wide[baseline], axis=0).drop(columns=[baseline])
    summary = pd.DataFrame(
        {
            "mean_edge": edge.mean(),
            "std_edge": edge.std(),
            "folds_ahead": (edge > 0).sum(),
            "n_folds": edge.notna().sum(),
        }
    )
    # The share of folds a model wins is a distribution-free read on whether an
    # edge is real. A model beating the baseline on 13 of 26 folds has not
    # beaten it, whatever the mean says.
    summary["win_rate"] = (summary["folds_ahead"] / summary["n_folds"]).round(3)
    return summary.sort_values("mean_edge", ascending=False).round(4)


def seed_spread(results: pd.DataFrame, metric: str = "directional_accuracy") -> pd.DataFrame:
    """Spread across random seeds, for the models that have one.

    If the seed to seed standard deviation exceeds a model's mean advantage
    over the baseline, the advantage has not been demonstrated. This table is
    what makes that comparison possible.
    """
    seeded = results.dropna(subset=["seed"])
    if seeded.empty or metric not in seeded.columns:
        return pd.DataFrame()

    per_fold = seeded.groupby(["model", "fold"])[metric].agg(["mean", "std"])
    return (
        per_fold.groupby("model")
        .agg(mean_across_seeds=("mean", "mean"), mean_seed_std=("std", "mean"))
        .round(4)
    )


def classify_regime(
    panel: pd.DataFrame,
    lookback_days: int = 90,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
) -> pd.Series:
    """Label each asset-date bull, bear or sideways by trailing return.

    The label uses only trailing data, so it is computable at the time and does
    not smuggle the future into the breakdown.
    """
    close = panel[schema.CLOSE]
    trailing = close.groupby(level=schema.ASSET, observed=True).transform(
        lambda s: s / s.shift(lookback_days) - 1.0
    )
    labels = pd.Series("sideways", index=panel.index, dtype="object")
    labels[trailing > bull_threshold] = "bull"
    labels[trailing < bear_threshold] = "bear"
    labels[trailing.isna()] = "unknown"
    return labels


def regime_breakdown(
    predictions: pd.DataFrame,
    y_true: pd.Series,
    regimes: pd.Series,
) -> pd.DataFrame:
    """Directional accuracy within each market regime.

    Regime dependence is the single most likely reason a model looks good on
    one sample and fails on the next, so it is worth isolating rather than
    averaging over.
    """
    if predictions.empty:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["y"] = y_true.reindex(frame.index)
    frame["regime"] = regimes.reindex(frame.index)
    frame = frame.dropna(subset=["y"])

    signal = frame["proba"] if "proba" in frame.columns else frame["pred"]
    frame["correct"] = ((signal >= 0.5).astype(float) == frame["y"]).astype(float)

    out = (
        frame.groupby(["model", "regime"], observed=True)
        .agg(accuracy=("correct", "mean"), base_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
    )
    out["edge_over_base"] = out["accuracy"] - out[["base_rate"]].assign(
        down=1 - out["base_rate"]
    ).max(axis=1)
    return out.round(4)


def per_asset_breakdown(predictions: pd.DataFrame, y_true: pd.Series) -> pd.DataFrame:
    """Directional accuracy per asset, next to that asset's own base rate."""
    if predictions.empty:
        return pd.DataFrame()

    frame = predictions.copy()
    frame["y"] = y_true.reindex(frame.index)
    frame = frame.dropna(subset=["y"])
    # Read the asset off the index rather than adding a column of the same
    # name, which pandas rejects as ambiguous inside groupby.
    asset_level = frame.index.get_level_values(schema.ASSET)

    signal = frame["proba"] if "proba" in frame.columns else frame["pred"]
    frame["correct"] = ((signal >= 0.5).astype(float) == frame["y"]).astype(float)

    return (
        frame.groupby(["model", asset_level.rename("asset")], observed=True)
        .agg(accuracy=("correct", "mean"), base_rate=("y", "mean"), n=("y", "size"))
        .reset_index()
        .round(4)
    )


PER_ASSET_SUFFIX = "_per_asset"


def scope_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    """Pooled fit against per-asset fit of the same family, on common rows only.

    A family run both ways appears as ``ridge`` and ``ridge_per_asset``. The
    per-asset arm can be missing rows the pooled arm has, because an asset with
    too little history is skipped rather than fitted on almost nothing, so the
    two are scored on the intersection of their rows and ``n_obs`` says how
    large that is. The Diebold-Mariano statistic is per-asset against pooled:
    negative means the per-asset fit has the lower loss.

    ``predictions`` must carry ``y_true``, which the runner writes next to the
    forecasts for exactly this purpose.
    """
    if predictions.empty or "y_true" not in predictions.columns:
        return pd.DataFrame()

    from src.evaluate.diebold_mariano import diebold_mariano

    models = set(predictions["model"].unique())
    pairs = [
        (name, name + PER_ASSET_SUFFIX)
        for name in sorted(models)
        if name + PER_ASSET_SUFFIX in models
    ]
    if not pairs:
        return pd.DataFrame()

    column = "proba" if "proba" in predictions.columns else "pred"
    metric = "accuracy" if column == "proba" else "rmse"
    usable = predictions.dropna(subset=[column])
    # Average a stochastic model over its seeds first, the same way dm_table and
    # the backtest do, so a comparison is between families and not between
    # lucky draws.
    signal = (
        usable.groupby([schema.ASSET, schema.DATE, "model"], observed=True)[column]
        .mean()
        .unstack("model")
    )
    truth = predictions["y_true"].groupby(level=[schema.ASSET, schema.DATE]).first()
    truth = truth.reindex(signal.index)
    assets = signal.index.get_level_values(schema.ASSET)

    rows = []
    for pooled, per_asset in pairs:
        if pooled not in signal.columns or per_asset not in signal.columns:
            # An arm that never produced a forecast has no column after the
            # NaN rows are dropped. There is nothing to compare it against.
            continue
        both = truth.notna() & signal[pooled].notna() & signal[per_asset].notna()
        for asset in [*sorted(assets.unique()), "all"]:
            mask = both if asset == "all" else both & (assets == asset)
            if int(mask.sum()) < 2:
                continue
            y = truth[mask].to_numpy()
            a = signal.loc[mask, pooled].to_numpy()
            b = signal.loc[mask, per_asset].to_numpy()
            result = diebold_mariano(y, b, a, loss="squared")
            if metric == "accuracy":
                score_a = float(np.mean((a >= 0.5) == (y == 1)))
                score_b = float(np.mean((b >= 0.5) == (y == 1)))
            else:
                score_a = float(np.sqrt(np.mean((a - y) ** 2)))
                score_b = float(np.sqrt(np.mean((b - y) ** 2)))
            rows.append(
                {
                    "family": pooled,
                    "asset": str(asset),
                    "n_obs": int(mask.sum()),
                    f"{metric}_pooled": round(score_a, 4),
                    f"{metric}_per_asset": round(score_b, 4),
                    "dm_statistic": result.statistic,
                    "p_value": result.p_value,
                    "better": {"a": "per_asset", "b": "pooled"}.get(result.better, result.better),
                }
            )
    return pd.DataFrame(rows)


def plot_accuracy_over_time(
    results: pd.DataFrame,
    baseline: str = "persistence",
    metric: str = "directional_accuracy",
    path: Path | None = None,
) -> Path | None:
    """Metric per fold with the baseline drawn as a reference line."""
    if metric not in results.columns or results.empty:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wide = results.pivot_table(index="test_start", columns="model", values=metric, aggfunc="mean")
    wide = wide.sort_index()

    figure, axis = plt.subplots(figsize=(11, 5))
    for column in wide.columns:
        style = "--" if column == baseline else "-"
        width = 2.2 if column == baseline else 1.4
        axis.plot(wide.index, wide[column], style, linewidth=width, label=column, marker="o", markersize=3)

    axis.axhline(0.5, color="black", linewidth=0.8, alpha=0.5)
    axis.set_ylabel(metric.replace("_", " "))
    axis.set_xlabel("test window start")
    axis.set_title(f"{metric.replace('_', ' ')} per walk-forward fold")
    axis.legend(fontsize=8, ncol=3)
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()

    path = path or (FIGURES_DIR / f"{metric}_by_fold.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_equity_curves(curves: dict[str, pd.Series], path: Path | None = None) -> Path | None:
    """Equity curves at several cost levels on one set of axes."""
    if not curves:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 5))
    for label, series in curves.items():
        style = "--" if "hold" in label.lower() else "-"
        axis.plot(series.index, series.to_numpy(), style, linewidth=1.5, label=label)

    axis.set_yscale("log")
    axis.set_ylabel("equity, log scale")
    axis.set_xlabel("date")
    axis.set_title("Strategy equity at each cost level, against buy and hold")
    axis.legend(fontsize=9)
    figure.tight_layout()

    path = path or (FIGURES_DIR / "equity_curves.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def write_markdown(run_dir: Path, sections: dict[str, pd.DataFrame], title: str) -> Path:
    """Write a results markdown file from a dict of named tables."""
    lines = [f"# {title}", "", f"Run `{run_dir.name}`.", ""]
    for heading, table in sections.items():
        if table is None or (isinstance(table, pd.DataFrame) and table.empty):
            continue
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(table.to_markdown())
        lines.append("")

    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise a completed run.")
    parser.add_argument("--run", default=None, help="run directory, defaults to the newest")
    parser.add_argument("--baseline", default="persistence")
    args = parser.parse_args()

    run_dir = Path(args.run) if args.run else latest_run()
    results, predictions = load_run(run_dir)

    # The runner writes dm.csv next to the results. Reading it back rather than
    # recomputing keeps the reported test identical to the one the run produced.
    dm_path = run_dir / "dm.csv"
    dm = pd.read_csv(dm_path) if dm_path.exists() else pd.DataFrame()
    dm_heading = f"Diebold-Mariano vs {dm['vs_baseline'].iloc[0]}" if not dm.empty else "Diebold-Mariano"
    if not dm.empty:
        # Indexed by model to match the other tables in the report.
        dm = dm.drop(columns=["vs_baseline"]).set_index("model")

    # Per-asset tables are written by the runner too, and read back for the
    # same reason as dm.csv: the report must show the numbers the run produced.
    per_asset = _read_optional(run_dir / "per_asset.csv")
    if not per_asset.empty:
        keep = [c for c in ["model", "asset", "n_folds", "n_obs", *CLASSIFICATION_COLUMNS,
                            "qlike", "mz_beta", "mz_r2", "rmse_log"] if c in per_asset.columns]
        per_asset = per_asset[keep].set_index(["model", "asset"]).round(4)
    per_asset_dm = _read_optional(run_dir / "dm_per_asset.csv")
    per_asset_dm_heading = (
        f"Diebold-Mariano vs {per_asset_dm['vs_baseline'].iloc[0]}, per asset"
        if not per_asset_dm.empty
        else "Diebold-Mariano, per asset"
    )
    if not per_asset_dm.empty:
        per_asset_dm = per_asset_dm.drop(columns=["vs_baseline", "note"], errors="ignore").set_index(
            ["model", "asset"]
        )
    scope = scope_comparison(predictions)
    if not scope.empty:
        scope = scope.set_index(["family", "asset"])

    sections = {
        "Per-fold summary": fold_table(results),
        f"Edge over {args.baseline}, per fold": edge_over_baseline(results, baseline=args.baseline),
        dm_heading: dm,
        "Seed spread": seed_spread(results),
        "By asset, rows pooled across folds and seeds": per_asset,
        per_asset_dm_heading: per_asset_dm,
        "Pooled fit vs per-asset fit, common rows": scope,
    }
    path = write_markdown(run_dir, sections, "Results")
    print(f"wrote {path}")

    # Name the figure after the run. Writing every run to the same filename
    # means the last command wins and a README pointing at the file silently
    # ends up describing a different experiment.
    figure = plot_accuracy_over_time(
        results,
        baseline=args.baseline,
        path=FIGURES_DIR / f"directional_accuracy_by_fold_{run_dir.name}.png",
    )
    if figure:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
