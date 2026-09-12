"""Turn a run's predictions into an equity curve and a cost sweep.

    python -m src.run_backtest --model lightgbm

Prediction accuracy is not profit, and the gap between the two is where most of
the interesting failure happens. A signal can be right slightly more often than
chance and still lose money because it trades constantly, or because it is
right on small moves and wrong on large ones.

The comparison that decides whether any of this was worth doing is buy and
hold on the same assets over the same dates, not a coin flip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import schema
from src.backtest.engine import buy_and_hold, cost_sweep, run_backtest, signal_to_position
from src.backtest.costs import CostModel
from src.config import load_config
from src.data.build_panel import load_auxiliary, load_panel
from src.data.targets import FWD_SIMPLE_RETURN
from src.evaluate.report import latest_run, load_run, plot_equity_curves
from src.evaluate.tables import seed_averaged
from src.features.registry import build_features


def forward_returns_for(index: pd.MultiIndex, config: dict) -> pd.Series:
    """Realised simple return over t to t+1 for every predicted row.

    Rebuilt from the frozen panel rather than carried through the run, so the
    backtest cannot be fed a return series that a model touched.
    """
    panel = load_panel()
    matrix = build_features(panel, config, load_auxiliary())
    return matrix.meta[FWD_SIMPLE_RETURN].reindex(index)


def positions_for(predictions: pd.DataFrame, model: str, config: dict) -> pd.Series:
    """Turn one model's predictions into positions under the configured rule."""
    subset = predictions[predictions["model"] == model]
    if subset.empty:
        raise ValueError(f"no predictions for model {model!r}")

    # Averaging across seeds first is the honest way to trade a stochastic
    # model: you would not get to pick the lucky seed in advance.
    signal_column = "proba" if "proba" in subset.columns else "pred"
    # Rows with no forecast stay in the book as flat days rather than dropping
    # out of the calendar, so turnover and buy and hold see the same dates.
    dates = subset.index.unique().sort_values()
    signal = seed_averaged(subset, signal_column)[model].reindex(dates)

    backtest_cfg = config["backtest"]
    _, periods_per_year = _backtest_settings(config)
    return signal_to_position(
        signal,
        rule=backtest_cfg.get("rule", "long_short"),
        threshold=backtest_cfg.get("threshold", 0.5),
        band=backtest_cfg.get("band", 0.02),
        periods_per_year=periods_per_year,
    )


def _backtest_settings(config: dict) -> tuple[list[float], int]:
    """Cost levels and calendar the sweep runs under, read once."""
    backtest_cfg = config["backtest"]
    levels = [float(x) for x in backtest_cfg.get("cost_bps_round_trip", [0.0, 5.0, 20.0])]
    return levels, int(backtest_cfg.get("periods_per_year", 365))


def per_asset_sweep(
    positions: pd.Series,
    forward: pd.Series,
    config: dict,
) -> pd.DataFrame:
    """The cost sweep run on each asset alone, as a standalone book.

    This is a different number from an asset's share of the equal-weight
    portfolio, which the engine also keeps. A reader who sees "BTC: 8%" will
    take it to mean trading BTC on its own, so that is what is computed: the
    engine's equal weighting reduces to unit positions when only one asset is
    present, and the buy-and-hold row is that one asset held.
    """
    levels, periods_per_year = _backtest_settings(config)
    assets = positions.index.get_level_values(schema.ASSET)
    frames = []
    for asset in sorted(assets.unique()):
        mask = assets == asset
        sweep = cost_sweep(positions[mask], forward[mask], levels, periods_per_year=periods_per_year)
        sweep.insert(0, "asset", str(asset))
        frames.append(sweep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def backtest_model(
    predictions: pd.DataFrame,
    model: str,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    """Cost sweep, equity curves and the per-asset sweep for one model."""
    positions = positions_for(predictions, model, config)

    forward = forward_returns_for(positions.index, config)
    keep = forward.notna()
    positions, forward = positions[keep], forward[keep]

    by_asset = per_asset_sweep(positions, forward, config)
    by_asset.insert(0, "model", model)

    levels, periods_per_year = _backtest_settings(config)
    sweep = cost_sweep(positions, forward, levels, periods_per_year=periods_per_year)
    sweep.insert(0, "model", model)

    curves: dict[str, pd.Series] = {}
    for level in levels:
        result = run_backtest(
            positions, forward, CostModel.from_round_trip_bps(level), periods_per_year=periods_per_year
        )
        curves[f"{model} at {level:g} bps"] = result.equity
    curves["buy and hold"] = buy_and_hold(forward, periods_per_year=periods_per_year).equity

    return sweep, curves, by_asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a run's predictions.")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--run", default=None, help="run directory, defaults to the newest")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="model name to backtest, repeatable. Defaults to every model in the run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = Path(args.run) if args.run else latest_run()
    _, predictions = load_run(run_dir)

    if predictions.empty:
        print(f"{run_dir} has no predictions to backtest")
        return

    models = args.model or sorted(predictions["model"].unique())
    sweeps, all_curves, asset_sweeps = [], {}, []

    for model in models:
        try:
            sweep, curves, by_asset = backtest_model(predictions, model, config)
        except Exception as error:
            print(f"  {model}: {error}")
            continue
        sweeps.append(sweep)
        asset_sweeps.append(by_asset)
        all_curves.update(curves)
        print(f"  {model}: backtested at {len(sweep) - 1} cost levels")

    if not sweeps:
        print("nothing backtested")
        return

    combined = pd.concat(sweeps, ignore_index=True)
    out_path = run_dir / "backtest.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")

    # Each asset traded on its own, next to the portfolio. The portfolio row
    # is not the average of these: an asset only carries weight on the days it
    # holds a position.
    per_asset = pd.concat(asset_sweeps, ignore_index=True)
    per_asset_path = run_dir / "backtest_per_asset.csv"
    per_asset.to_csv(per_asset_path, index=False)
    print(f"wrote {per_asset_path}")

    columns = [
        c
        for c in ["model", "cost_bps", "annual_return", "sharpe", "max_drawdown", "annual_turnover", "hit_rate"]
        if c in combined.columns
    ]
    print(combined[columns].round(4).to_string(index=False))

    figure = plot_equity_curves(all_curves)
    if figure:
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
