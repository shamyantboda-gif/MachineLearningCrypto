"""Signal to position mapping, the backtest loop, and its metrics.

Written directly against pandas and numpy rather than a backtesting library.
The whole point of this layer is that there is no ambiguity about what it does
with a signal, and a dependency that hides the alignment convention would
defeat that.

Two conventions are load bearing and are repeated wherever they apply:

1. The position at date ``t`` is decided from information available at ``t``
   and earns ``forward_returns[t]``, the simple return realised from ``t`` to
   ``t + 1``. No shifting happens inside this module. The caller is
   responsible for handing over a forward return that is genuinely in the
   future relative to the features that produced the signal.
2. Turnover at ``t`` is ``abs(position_t - position_{t-1})`` per asset, so a
   round trip registers 2.0 and is charged twice. See ``costs.CostModel``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..schema import ASSET, DATE
from .costs import CostModel

__all__ = [
    "BacktestResult",
    "signal_to_position",
    "run_backtest",
    "compute_stats",
    "buy_and_hold",
    "cost_sweep",
]

# Crypto trades every day of the year, so a year is 365 periods on a daily
# panel. The 252 used for equities counts exchange business days and would
# overstate an annualised crypto Sharpe by sqrt(365/252), about 20 percent,
# purely as an artefact of the calendar.
PERIODS_PER_YEAR = 365

STAT_KEYS = [
    "total_return",
    "annual_return",
    "annual_vol",
    "sharpe",
    "sortino",
    "max_drawdown",
    "max_drawdown_days",
    "max_drawdown_unrecovered",
    "calmar",
    "hit_rate",
    "avg_win",
    "avg_loss",
    "win_loss_ratio",
    "annual_turnover",
    "n_periods",
    "exposure",
]


# --------------------------------------------------------------------------
# index plumbing
# --------------------------------------------------------------------------


def _check_panel_series(series: pd.Series, name: str) -> pd.Series:
    """Return ``series`` as a float Series on a sorted (asset, date) index."""
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series, got {type(series).__name__}")
    if not isinstance(series.index, pd.MultiIndex) or series.index.nlevels != 2:
        raise ValueError(
            f"{name} must carry a two level ({ASSET}, {DATE}) MultiIndex, "
            f"got index names {list(series.index.names)}"
        )
    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()][:5].tolist()
        raise ValueError(f"{name} has duplicate (asset, date) rows, first few: {dupes}")
    out = series.astype("float64").sort_index()
    out.name = name
    return out


def _empty_result(label: str, cost_model: CostModel, periods_per_year: int) -> "BacktestResult":
    """A result object for an empty overlap, so callers never have to branch."""
    index = pd.MultiIndex.from_arrays([[], []], names=[ASSET, DATE])
    empty_dates = pd.Series(dtype="float64")
    per_asset = pd.DataFrame(
        {
            "position": pd.Series(dtype="float64"),
            "forward_return": pd.Series(dtype="float64"),
            "gross": pd.Series(dtype="float64"),
            "turnover": pd.Series(dtype="float64"),
            "cost": pd.Series(dtype="float64"),
            "net": pd.Series(dtype="float64"),
            "weight": pd.Series(dtype="float64"),
            "weight_turnover": pd.Series(dtype="float64"),
        },
        index=index,
    )
    return BacktestResult(
        gross_returns=empty_dates,
        net_returns=empty_dates,
        equity=empty_dates,
        turnover=empty_dates,
        costs=empty_dates,
        n_active=empty_dates,
        per_asset=per_asset,
        stats=compute_stats(empty_dates, periods_per_year=periods_per_year),
        label=label,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
    )


# --------------------------------------------------------------------------
# signal to position
# --------------------------------------------------------------------------


def signal_to_position(
    proba_up: pd.Series,
    rule: str = "long_short",
    threshold: float = 0.5,
    band: float = 0.02,
    vol_forecast: pd.Series | None = None,
    annual_target: float | None = None,
    max_leverage: float = 1.0,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.Series:
    """Map a probability of an up move to a target position.

    Parameters
    ----------
    proba_up:
        P(up) for each (asset, date), as produced by a classification model.
        The position it yields is the position held into that date's forward
        return, so ``proba_up`` must itself have been computed from data at or
        before that date.
    rule:
        ``long_only`` goes +1 when ``proba_up > threshold`` and flat otherwise.
        ``long_short`` goes +1 above ``0.5 + band``, -1 below ``0.5 - band``,
        and flat inside the band. The dead zone exists so that a model hovering
        around a coin flip does not churn the book.
    vol_forecast:
        Per period standard deviation of simple returns, on the same scale as
        the returns the backtest will use. A daily panel means a daily standard
        deviation, so 0.03 is a 3 percent daily move, not 3 percent annualised.
        Supply it together with ``annual_target`` to size positions inversely
        to forecast risk.
    annual_target:
        Target annualised volatility of the position, for example 0.40 for 40
        percent. Ignored unless ``vol_forecast`` is also given.
    max_leverage:
        Cap on the absolute position size after scaling.

    Returns
    -------
    A float Series on the same (asset, date) index, sorted.
    """
    proba = _check_panel_series(proba_up, "proba_up")

    if rule == "long_only":
        raw = (proba > threshold).astype("float64")
    elif rule == "long_short":
        raw = np.where(
            proba > 0.5 + band,
            1.0,
            np.where(proba < 0.5 - band, -1.0, 0.0),
        )
        raw = pd.Series(raw, index=proba.index, dtype="float64")
    else:
        raise ValueError(f"unknown rule {rule!r}, expected 'long_only' or 'long_short'")

    # A missing probability is not a neutral probability. Treat it as no view
    # and stay flat, rather than letting NaN comparisons quietly produce 0.
    raw = raw.where(proba.notna(), 0.0)

    if vol_forecast is None or annual_target is None:
        position = raw
    else:
        vol = _check_panel_series(vol_forecast, "vol_forecast").reindex(proba.index)
        # Annualise the per period standard deviation before comparing it to an
        # annual target, otherwise the scale factor is off by sqrt(365).
        annual_vol = vol * np.sqrt(float(periods_per_year))
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = float(annual_target) / annual_vol
        # A zero, negative, missing or infinite volatility forecast carries no
        # sizing information. Sizing off it would either divide by zero or take
        # unbounded leverage on the asset the model understands least, so those
        # rows go flat.
        scale = scale.where(np.isfinite(scale) & (annual_vol > 0.0), 0.0)
        sized = raw * scale
        position = np.sign(sized) * sized.abs().clip(lower=0.0, upper=float(max_leverage))

    position = position.astype("float64")
    position.name = "position"
    return position.sort_index()


# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Everything one backtest produced, at both the portfolio and asset level.

    The portfolio level series are indexed by date. ``per_asset`` keeps the
    (asset, date) detail so that an unexpected number can be traced back to the
    position and forward return that produced it. Its ``turnover``, ``cost``
    and ``net`` columns describe each asset as a standalone book at full size,
    while ``weight`` and ``weight_turnover`` are that asset's contribution to
    the equal weighted portfolio, which is what the date level series are built
    from.
    """

    gross_returns: pd.Series
    net_returns: pd.Series
    equity: pd.Series
    turnover: pd.Series
    costs: pd.Series
    n_active: pd.Series
    per_asset: pd.DataFrame
    stats: dict[str, float] = field(default_factory=dict)
    label: str = "strategy"
    cost_model: CostModel | None = None
    periods_per_year: int = PERIODS_PER_YEAR

    def summary(self) -> pd.DataFrame:
        """One row of stats, indexed by the run label, for stacking in tables."""
        row = {key: self.stats.get(key, np.nan) for key in STAT_KEYS}
        frame = pd.DataFrame([row], index=pd.Index([self.label], name="run"))
        return frame

    def __str__(self) -> str:
        stats = self.stats
        return (
            f"{self.label}: total {stats.get('total_return', np.nan):.4f} "
            f"CAGR {stats.get('annual_return', np.nan):.4f} "
            f"Sharpe {stats.get('sharpe', np.nan):.3f} "
            f"maxDD {stats.get('max_drawdown', np.nan):.4f} "
            f"over {int(stats.get('n_periods', 0) or 0)} periods"
        )


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------


def run_backtest(
    positions: pd.Series,
    forward_returns: pd.Series,
    cost_model: CostModel,
    periods_per_year: int = PERIODS_PER_YEAR,
    label: str = "strategy",
) -> BacktestResult:
    """Run one backtest over a (asset, date) panel of positions.

    Alignment, which is the only thing in this file that can silently destroy a
    result: the position at date ``t`` is the position decided from information
    available at ``t``, and it earns ``forward_returns[t]``, the simple return
    realised from ``t`` to ``t + 1``. So the per asset gross return is

        gross_t = position_t * forward_return_t

    with no shift applied here. Shifting inside the engine as well as inside the
    target construction is the classic way to end up either one day late (which
    destroys a real edge) or one day early (which invents one).

    Turnover at ``t`` is ``abs(position_t - position_{t-1})`` computed per
    asset, with the first observation of each asset treated as arriving from a
    flat book, so opening the initial position is charged. Cost is
    ``cost_rate * turnover_t`` and ``net_t = gross_t - cost_t``.

    Portfolio construction is equal weight across whichever assets carry a
    position on that date, so each date's weights sum to one in absolute terms
    when every active asset is fully sized. That keeps a single asset run and a
    four asset run on comparable scales: without it, summing four positions of
    +1 would quietly run four times the leverage and produce a Sharpe that is
    not comparable to the single asset case. Portfolio turnover and cost are
    then computed on those weights, which is what charges the exit of a
    position and the rebalance of the survivors when the active set changes.

    Rows where the forward return is missing are dropped before anything is
    computed, since the last date of each asset has no realised next day
    return. The count that was dropped is recorded in the result stats.
    """
    pos = _check_panel_series(positions, "position")
    fwd = _check_panel_series(forward_returns, "forward_return")

    frame = pd.concat([pos, fwd], axis=1, join="inner").sort_index()
    n_before = len(frame)
    frame = frame[np.isfinite(frame["forward_return"].to_numpy())]
    n_dropped = n_before - len(frame)
    # A missing position is no view, which is flat, not a NaN that poisons the
    # whole portfolio return for that date.
    frame["position"] = frame["position"].fillna(0.0)

    if frame.empty:
        result = _empty_result(label, cost_model, periods_per_year)
        result.stats["n_rows_dropped_no_forward_return"] = float(n_dropped)
        return result

    frame["gross"] = frame["position"] * frame["forward_return"]

    # Previous position within each asset. The first observation of an asset
    # has no predecessor, and a book that has never traded is flat, so the
    # missing predecessor is 0.0 and the opening trade gets charged.
    prev = frame["position"].groupby(level=0, sort=False).shift(1).fillna(0.0)
    frame["turnover"] = (frame["position"] - prev).abs()
    frame["cost"] = cost_model.cost_for_turnover(frame["turnover"])
    frame["net"] = frame["gross"] - frame["cost"]

    # Equal weight across the assets that actually hold risk on that date. An
    # asset that is flat contributes no weight, so it cannot dilute the assets
    # that are positioned: two assets long and one flat is a two way split, not
    # a three way one.
    n_active = (frame["position"] != 0.0).astype("float64").groupby(level=1, sort=True).sum()
    denom = n_active.reindex(frame.index, level=1)
    frame["weight"] = (frame["position"] / denom.where(denom > 0.0)).fillna(0.0)

    # Portfolio turnover is measured on the weights rather than averaged from
    # the per asset turnover. Averaging would drop the exit trade of an asset
    # that is flat by the end of the date, and it would also miss the rebalance
    # that genuinely happens when the active set changes and every surviving
    # position has to be resized. Both are real trades that pay real fees. With
    # a single asset the weight is the position, so this reduces exactly to
    # abs(position_t - position_{t-1}).
    prev_weight = frame["weight"].groupby(level=0, sort=False).shift(1).fillna(0.0)
    frame["weight_turnover"] = (frame["weight"] - prev_weight).abs()

    gross_returns = (
        (frame["weight"] * frame["forward_return"])
        .groupby(level=1, sort=True)
        .sum()
        .rename("gross_return")
    )
    turnover = frame["weight_turnover"].groupby(level=1, sort=True).sum().rename("turnover")
    costs = cost_model.cost_for_turnover(turnover).rename("cost")
    net_returns = (gross_returns - costs).rename("net_return")

    equity = (1.0 + net_returns).cumprod().rename("equity")

    held = (frame["position"] != 0.0).groupby(level=1, sort=True).any()
    stats = compute_stats(
        net_returns,
        turnover=turnover,
        exposure_mask=held,
        periods_per_year=periods_per_year,
    )
    stats["n_rows_dropped_no_forward_return"] = float(n_dropped)
    stats["n_assets"] = float(frame.index.get_level_values(0).nunique())

    return BacktestResult(
        gross_returns=gross_returns,
        net_returns=net_returns,
        equity=equity,
        turnover=turnover,
        costs=costs,
        n_active=n_active.rename("n_active"),
        per_asset=frame,
        stats=stats,
        label=label,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
    )


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def _drawdown_days(equity: pd.Series) -> tuple[float, bool]:
    """Longest peak to recovery span, in days, and whether it is still open.

    The clock starts at the first date with capital at 1.0, so a strategy that
    loses money immediately is underwater from the first observation. If the
    equity curve never regains its last peak, the span from that peak to the
    end of the sample is reported and the flag is True, because the honest
    answer there is "at least this long, we do not know", not the length of
    some earlier drawdown that did recover.
    """
    if equity.empty:
        return float("nan"), False

    index = equity.index
    is_datetime = isinstance(index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(index)
    values = equity.to_numpy(dtype="float64")
    if not np.isfinite(values).any():
        return float("nan"), False

    def _span(start_pos: int, end_pos: int) -> float:
        if is_datetime:
            delta = pd.Timestamp(index[end_pos]) - pd.Timestamp(index[start_pos])
            return float(delta.days)
        # Without a date index the best available unit is periods elapsed.
        return float(end_pos - start_pos)

    peak_value = 1.0
    peak_pos = 0
    longest = 0.0
    # A span only counts once the curve has actually dipped below its peak.
    # Without this flag an equity curve that only ever grinds upward would
    # report a drawdown as long as its own longest flat stretch.
    underwater = False
    for pos in range(len(values)):
        value = values[pos]
        if not np.isfinite(value):
            continue
        if value >= peak_value:
            if underwater:
                longest = max(longest, _span(peak_pos, pos))
            peak_value = value
            peak_pos = pos
            underwater = False
        else:
            underwater = True

    open_at_end = False
    if underwater:
        tail = _span(peak_pos, len(values) - 1)
        if tail >= longest:
            # The reported span is the one still open, so the honest reading is
            # "at least this long". An earlier, longer, fully recovered
            # drawdown wins the number and clears the flag, otherwise the flag
            # would describe a span it is not attached to.
            longest = tail
            open_at_end = True
    return float(longest), open_at_end


def compute_stats(
    net_returns: pd.Series,
    turnover: pd.Series | None = None,
    exposure_mask: pd.Series | None = None,
    periods_per_year: int = PERIODS_PER_YEAR,
) -> dict[str, float]:
    """Performance statistics for one per period net return series.

    Every statistic returns nan rather than raising when the input cannot
    support it, so an empty fold, an all flat strategy, or a strategy that
    never wins still produces a complete row in the results table. A results
    table with a hole in it is easier to read than a traceback in the middle of
    a sweep.

    ``exposure_mask`` is a boolean per date saying whether any capital was at
    risk. Hit rate is measured only over those dates, because counting flat
    days as losses would punish a selective strategy for staying out.
    """
    nan = float("nan")
    stats: dict[str, float] = {key: nan for key in STAT_KEYS}
    stats["max_drawdown_unrecovered"] = False

    returns = pd.Series(net_returns, dtype="float64").dropna()
    n = int(len(returns))
    stats["n_periods"] = float(n)
    if n == 0:
        stats["exposure"] = nan
        return stats

    values = returns.to_numpy(dtype="float64")

    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    stats["total_return"] = total_return

    years = n / float(periods_per_year)
    if years > 0.0 and (1.0 + total_return) > 0.0:
        stats["annual_return"] = float((1.0 + total_return) ** (1.0 / years) - 1.0)
    # A terminal equity of zero or below means the account was wiped out. There
    # is no real compound growth rate for that, so it stays nan.

    if n > 1:
        vol = float(np.std(values, ddof=1))
        stats["annual_vol"] = vol * float(np.sqrt(periods_per_year))
        mean = float(np.mean(values))
        if vol > 0.0:
            # Excess return over a zero risk free rate. Annualising with 365
            # rather than 252: crypto has no weekends, so 365 periods really do
            # elapse in a year and using 252 would inflate this by about 20
            # percent.
            stats["sharpe"] = mean / vol * float(np.sqrt(periods_per_year))
        downside = np.minimum(values, 0.0)
        downside_dev = float(np.sqrt(np.mean(downside**2)))
        if downside_dev > 0.0:
            stats["sortino"] = mean / downside_dev * float(np.sqrt(periods_per_year))

    # The peak the drawdown is measured from starts at the initial capital of
    # 1.0, not at the first date's equity, so a loss on day one counts as a
    # drawdown instead of being absorbed into a new running maximum.
    running_max = equity.cummax().clip(lower=1.0)
    drawdown = equity / running_max - 1.0
    stats["max_drawdown"] = float(drawdown.min()) if len(drawdown) else nan

    dd_days, unrecovered = _drawdown_days(equity)
    stats["max_drawdown_days"] = dd_days
    stats["max_drawdown_unrecovered"] = unrecovered

    max_dd = stats["max_drawdown"]
    annual_return = stats["annual_return"]
    if (
        np.isfinite(annual_return)
        and np.isfinite(max_dd)
        and max_dd < 0.0
    ):
        stats["calmar"] = float(annual_return / abs(max_dd))

    if exposure_mask is None:
        active = pd.Series(True, index=returns.index)
    else:
        active = pd.Series(exposure_mask).reindex(returns.index).fillna(False).astype(bool)
    n_active = int(active.sum())
    stats["exposure"] = float(n_active) / float(n)

    live = returns[active.to_numpy()]
    if len(live):
        wins = live[live > 0.0]
        losses = live[live < 0.0]
        stats["hit_rate"] = float(len(wins)) / float(len(live))
        if len(wins):
            stats["avg_win"] = float(wins.mean())
        if len(losses):
            stats["avg_loss"] = float(losses.mean())
        if len(wins) and len(losses) and float(losses.mean()) != 0.0:
            stats["win_loss_ratio"] = float(abs(wins.mean() / losses.mean()))

    if turnover is not None:
        turn = pd.Series(turnover, dtype="float64").reindex(returns.index).dropna()
        if len(turn):
            stats["annual_turnover"] = float(turn.mean()) * float(periods_per_year)

    return stats


# --------------------------------------------------------------------------
# comparisons
# --------------------------------------------------------------------------


def buy_and_hold(
    forward_returns: pd.Series,
    periods_per_year: int = PERIODS_PER_YEAR,
    label: str = "buy_and_hold",
) -> BacktestResult:
    """Always long, equal weight, evaluated on exactly the strategy's dates.

    This, not a coin flip, is the benchmark that matters. Over most crypto
    windows simply holding the asset has a strong Sharpe, so a directional
    model that beats 50 percent accuracy has demonstrated nothing until it also
    beats holding on a risk adjusted basis. Pass the same ``forward_returns``
    the strategy was run on so the comparison covers the same dates and the
    same assets.

    Turnover is 1.0 on each asset's first date and zero thereafter, and costs
    are set to zero. A single entry commission spread over a multi year hold is
    a rounding error, and a benchmark that moves with the strategy's assumed
    fee schedule is harder to reason about than a fixed reference line.
    """
    fwd = _check_panel_series(forward_returns, "forward_return")
    positions = pd.Series(1.0, index=fwd.index, name="position")
    return run_backtest(
        positions,
        fwd,
        cost_model=CostModel(commission_bps_round_trip=0.0, slippage_bps_per_trade=0.0),
        periods_per_year=periods_per_year,
        label=label,
    )


def cost_sweep(
    positions: pd.Series,
    forward_returns: pd.Series,
    bps_levels: list[float],
    periods_per_year: int = PERIODS_PER_YEAR,
) -> pd.DataFrame:
    """Run the same strategy at several round trip cost levels.

    Each level is the total cost of a full round trip in basis points, matching
    ``backtest.cost_bps_round_trip`` in the config. The buy and hold row is
    appended for reference and carries a nan cost level, since it is a
    benchmark rather than another point on the cost axis.

    An edge that survives 0 bps and dies at 5 bps is a cost problem, not a
    strategy, and the whole reason this returns every level in one frame is to
    make that visible in a single glance.
    """
    rows: list[pd.DataFrame] = []
    for bps in bps_levels:
        model = CostModel.from_round_trip_bps(float(bps))
        result = run_backtest(
            positions,
            forward_returns,
            cost_model=model,
            periods_per_year=periods_per_year,
            label=f"strategy_{float(bps):g}bps",
        )
        row = result.summary()
        row.insert(0, "cost_bps", float(bps))
        row.insert(1, "run_type", "strategy")
        rows.append(row)

    bh = buy_and_hold(forward_returns, periods_per_year=periods_per_year)
    bh_row = bh.summary()
    bh_row.insert(0, "cost_bps", np.nan)
    bh_row.insert(1, "run_type", "buy_and_hold")
    rows.append(bh_row)

    return pd.concat(rows, axis=0)
