"""The target alignment trap.

The whole project rests on one convention: a row indexed ``t`` holds features
observed at ``t`` and a target that is a fact about ``t+1``. If the target is
shifted the wrong way the row indexed ``t`` holds a target from ``t-1``, the
model learns to read the answer off its own features, and every metric in the
report becomes meaningless while looking excellent.

Row-by-row checks run on the five-day hand-built panel where the right answer
can be written down. The correlation checks run on the 900-day synthetic panel
because ``corr(close[t], close[t+1])`` is only near 0.999 over a long series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import schema
from src.data import targets as targets_module
from src.data.targets import DIR, FWD_SIMPLE_RETURN, RET, build_targets
from src.features.registry import build_features
from tests.conftest import requires_real_panel


# ---------------------------------------------------------------------------
# Row by row against hand computed answers
# ---------------------------------------------------------------------------


def test_direction_target_matches_hand_computed_answer(tiny_panel):
    """dir_1d at t is 1 exactly when close[t+1] > close[t].

    Catches: a target shifted the wrong way, an off-by-one, and a flat day
    being scored as up. The flat BTC day (105 -> 105) must be 0, because a
    strictly-greater comparison is what the backtest's long rule assumes.
    """
    result = build_targets(tiny_panel)

    expected = {
        ("BTC", "2020-01-01"): 1.0,  # 100 -> 110
        ("BTC", "2020-01-02"): 0.0,  # 110 -> 105
        ("BTC", "2020-01-03"): 0.0,  # 105 -> 105, flat is not up
        ("BTC", "2020-01-04"): 1.0,  # 105 -> 120
        ("ETH", "2020-01-01"): 0.0,  # 1000 -> 900
        ("ETH", "2020-01-02"): 1.0,  # 900 -> 950
        ("ETH", "2020-01-03"): 1.0,  # 950 -> 1000
        ("ETH", "2020-01-04"): 1.0,  # 1000 -> 1100
    }

    for (asset, day), want in expected.items():
        key = (asset, pd.Timestamp(day, tz="UTC"))
        got = result.loc[key, DIR]
        assert got == want, (
            f"dir_1d at {asset} {day} is {got}, expected {want}. "
            "The target is not aligned to t+1."
        )


def test_log_return_target_matches_hand_computed_answer(tiny_panel):
    """ret_1d at t equals log(close[t+1]) - log(close[t]).

    Catches a target built as a backward return, which would make ret_1d at t
    equal log(close[t]) - log(close[t-1]) and be perfectly predictable from a
    feature the model already has (px_ret_1).
    """
    result = build_targets(tiny_panel)
    close = tiny_panel[schema.CLOSE]

    for asset in ("BTC", "ETH"):
        series = close.xs(asset, level=schema.ASSET)
        for position in range(len(series) - 1):
            key = (asset, series.index[position])
            want = float(np.log(series.iloc[position + 1]) - np.log(series.iloc[position]))
            got = float(result.loc[key, RET])
            assert got == pytest.approx(want, abs=1e-12), (
                f"ret_1d at {key} is {got}, expected {want}"
            )


def test_forward_simple_return_matches_hand_computed_answer(tiny_panel):
    """fwd_ret_simple at t equals close[t+1]/close[t] - 1.

    This is the column the backtest turns into PnL. If it were shifted the
    other way the backtest would pay yesterday's move for today's position and
    report a Sharpe ratio that cannot be traded.
    """
    result = build_targets(tiny_panel)
    close = tiny_panel[schema.CLOSE]

    for asset in ("BTC", "ETH"):
        series = close.xs(asset, level=schema.ASSET)
        for position in range(len(series) - 1):
            key = (asset, series.index[position])
            want = float(series.iloc[position + 1] / series.iloc[position] - 1.0)
            got = float(result.loc[key, FWD_SIMPLE_RETURN])
            assert got == pytest.approx(want, abs=1e-12), (
                f"fwd_ret_simple at {key} is {got}, expected {want}"
            )


def test_last_row_of_each_asset_has_a_nan_target(tiny_panel):
    """The final bar of every asset has no t+1, so every target must be NaN.

    Catches a fillna or a ffill sneaking into the target code, which would
    invent a fake observation at the end of each asset's history.
    """
    result = build_targets(tiny_panel)

    for asset in ("BTC", "ETH"):
        last_date = tiny_panel.xs(asset, level=schema.ASSET).index.max()
        row = result.loc[(asset, last_date)]
        for column in (RET, DIR, FWD_SIMPLE_RETURN):
            assert pd.isna(row[column]), (
                f"{column} at the last {asset} bar ({last_date.date()}) is "
                f"{row[column]!r}, expected NaN. A target was invented for a "
                "bar that has no successor."
            )


def test_registry_drops_the_final_bar_of_each_asset(synthetic_panel, base_config):
    """The NaN-target rows must actually leave the feature matrix.

    Catches the case where build_targets is correct but the registry keeps the
    row anyway, handing a model a NaN label it may quietly impute to zero.
    """
    config = base_config
    config["features"]["max_lookback"] = 30
    matrix = build_features(synthetic_panel, config)

    for asset in synthetic_panel.index.get_level_values(schema.ASSET).unique():
        last_date = synthetic_panel.xs(asset, level=schema.ASSET).index.max()
        assert (asset, last_date) not in matrix.X.index, (
            f"the final {asset} bar {last_date.date()} survived build_features "
            "even though its target is NaN"
        )

    # And nothing that did survive carries a NaN primary target.
    primary = config["targets"]["primary"]
    assert matrix.targets[primary].notna().all(), (
        "build_features kept rows whose primary target is NaN"
    )


# ---------------------------------------------------------------------------
# Per asset construction
# ---------------------------------------------------------------------------


def test_targets_are_computed_per_asset_not_across_the_seam(tiny_panel):
    """BTC's last row must not take its target from ETH's first row.

    The panel is sorted by (asset, date), so BTC's last row sits immediately
    before ETH's first row. A plain ``close.shift(-1)`` with no groupby would
    give BTC's last bar a target drawn from ETH's opening price. The tiny panel
    puts ETH an order of magnitude above BTC precisely so that bug produces a
    return of about +2.3 instead of a plausible small number.
    """
    result = build_targets(tiny_panel)
    close = tiny_panel[schema.CLOSE]

    btc_last = tiny_panel.xs("BTC", level=schema.ASSET).index.max()

    # Positive control: the buggy, groupby-free version really does produce a
    # value here. Without this the assertion below could pass for the wrong
    # reason, for instance if the seam were not where we think it is.
    naive = np.log(close).shift(-1) - np.log(close)
    seam_value = naive.loc[("BTC", btc_last)]
    assert not pd.isna(seam_value), (
        "the test is broken: the groupby-free version produced NaN at the "
        "BTC/ETH seam, so this test cannot detect a missing groupby"
    )
    assert seam_value > 1.0, (
        f"the test is broken: the seam leak is only {seam_value:.4f}, too small "
        "to be distinguishable from a real return"
    )

    # The real assertion: the shipped code must not produce that value.
    seam_ret = result.loc[("BTC", btc_last), RET]
    assert pd.isna(seam_ret), (
        f"ret_1d at the last BTC bar is {seam_ret!r}, expected NaN. The target "
        "shift is reaching across the asset boundary into ETH."
    )
    assert pd.isna(result.loc[("BTC", btc_last), FWD_SIMPLE_RETURN]), (
        "fwd_ret_simple at the last BTC bar reached across into ETH"
    )
    assert pd.isna(result.loc[("BTC", btc_last), DIR]), (
        "dir_1d at the last BTC bar reached across into ETH"
    )

    # Nothing anywhere in the frame should carry the seam magnitude either.
    finite = result[RET].dropna()
    assert finite.abs().max() < 1.0, (
        f"largest |ret_1d| is {finite.abs().max():.4f}; a cross-asset shift is "
        "leaking a price-level jump into the return target"
    )


# ---------------------------------------------------------------------------
# The shift direction test
# ---------------------------------------------------------------------------

# A genuine next-day return target is close to unpredictable from the current
# price level: on both the synthetic and the real panel |corr(close, target)|
# comes out below 0.1. A target that was shifted the wrong way, or left as a
# price level instead of a return, correlates about 0.999 with close because
# consecutive daily closes are almost the same number. 0.5 therefore sits in
# the empty middle of those two regimes: far enough above the honest value to
# never trip on noise, far enough below 0.999 to catch the bug outright.
MAX_CLOSE_TARGET_CORR = 0.5


def _close_target_correlations(panel: pd.DataFrame) -> dict[str, float]:
    result = build_targets(panel)
    close = panel[schema.CLOSE]
    return {
        column: float(np.abs(close.corr(result[column])))
        for column in (RET, DIR, FWD_SIMPLE_RETURN)
    }


def test_close_is_not_correlated_with_its_own_target(synthetic_panel):
    """corr(close[t], target[t]) must not be near 1.

    This is the single test most likely to catch a wrong-way shift. See the
    comment on MAX_CLOSE_TARGET_CORR for why the threshold is 0.5.
    """
    correlations = _close_target_correlations(synthetic_panel)

    # Positive control first: prove that the bug this test hunts for would
    # actually show up as a near-1 correlation on this data. If the control
    # does not fire, the assertion below proves nothing.
    close = synthetic_panel[schema.CLOSE]
    wrong_way = close.groupby(level=schema.ASSET, observed=True).shift(-1)
    control = float(np.abs(close.corr(wrong_way)))
    assert control > 0.99, (
        f"the test is broken: a deliberately mis-specified level target only "
        f"reaches |corr| = {control:.4f} on this panel, so a real wrong-way "
        "shift might slip past the 0.5 threshold"
    )

    offenders = {k: v for k, v in correlations.items() if v > MAX_CLOSE_TARGET_CORR}
    assert not offenders, (
        f"close is correlated with its own target: {offenders}. "
        f"For reference a deliberately wrong-way target scores {control:.4f} "
        "on this panel. The target is almost certainly shifted the wrong way "
        "or is a price level rather than a return."
    )


def test_target_is_correlated_with_the_next_close_not_the_current_one(synthetic_panel):
    """The forward return must line up with tomorrow's close, not today's.

    A one-sided version of the test above: fwd_ret_simple at t should track the
    forward relative change close[t+1]/close[t] - 1 almost exactly, and should
    be near uncorrelated with the backward one close[t]/close[t-1] - 1. If those
    two swap places the target is lagged by a bar.

    Both comparisons are built as relative changes, not price differences. An
    absolute difference is not comparable across assets trading at different
    price levels, and mixing the two would dilute the correlation for reasons
    that have nothing to do with alignment.
    """
    result = build_targets(synthetic_panel)
    close = synthetic_panel[schema.CLOSE]
    by_asset = close.groupby(level=schema.ASSET, observed=True)

    forward_change = by_asset.shift(-1) / close - 1.0
    backward_change = close / by_asset.shift(1) - 1.0

    forward_corr = float(result[FWD_SIMPLE_RETURN].corr(forward_change))
    backward_corr = float(result[FWD_SIMPLE_RETURN].corr(backward_change))

    assert forward_corr > 0.999, (
        f"fwd_ret_simple correlates only {forward_corr:.4f} with the actual "
        "t to t+1 relative price change; the target is not measuring the future"
    )
    assert forward_corr > abs(backward_corr) + 0.5, (
        f"fwd_ret_simple correlates {forward_corr:.4f} with the forward change "
        f"and {backward_corr:.4f} with the backward change. The target is "
        "lagged by one bar."
    )


def test_targets_module_is_the_only_place_that_shifts_backwards():
    """Guard the assumption the leakage grep in test_no_leakage.py relies on.

    That grep allowlists exactly one file path. If the target code ever moves,
    the allowlist silently starts protecting a file that no longer exists and
    the grep stops being a real constraint.
    """
    source = targets_module.__file__.replace("\\", "/")
    assert source.endswith("src/data/targets.py"), (
        f"targets module now lives at {source}; update the negative-shift "
        "allowlist in tests/test_no_leakage.py to match"
    )


# ---------------------------------------------------------------------------
# Real panel checks. Skipped when data/interim/panel_1d.parquet is absent.
# ---------------------------------------------------------------------------


@requires_real_panel
def test_real_panel_direction_target_matches_close_comparison(real_panel):
    """Row by row on the real panel: dir_1d == 1 iff close[t+1] > close[t]."""
    result = build_targets(real_panel)
    close = real_panel[schema.CLOSE]
    next_close = close.groupby(level=schema.ASSET, observed=True).shift(-1)

    expected = (next_close > close).astype("float64").where(next_close.notna())
    got = result[DIR]

    both = expected.notna() & got.notna()
    mismatches = both & (expected != got)
    assert not mismatches.any(), (
        f"{int(mismatches.sum())} rows where dir_1d disagrees with "
        f"close[t+1] > close[t]; first few: "
        f"{result.index[mismatches][:5].tolist()}"
    )
    # The NaN pattern must match too, otherwise a row was invented or lost.
    assert (expected.isna() == got.isna()).all(), (
        "dir_1d is NaN on a different set of rows than close[t+1] is"
    )


@requires_real_panel
def test_real_panel_last_row_per_asset_is_nan_and_dropped(real_panel, base_config):
    """Every asset's final bar must be NaN in targets and absent after build."""
    result = build_targets(real_panel)
    matrix = build_features(real_panel, base_config)

    for asset in real_panel.index.get_level_values(schema.ASSET).unique():
        last_date = real_panel.xs(asset, level=schema.ASSET).index.max()
        assert pd.isna(result.loc[(asset, last_date), RET]), (
            f"real panel: ret_1d at the last {asset} bar is not NaN"
        )
        assert (asset, last_date) not in matrix.X.index, (
            f"real panel: the last {asset} bar survived build_features"
        )


@requires_real_panel
def test_real_panel_close_is_not_correlated_with_its_own_target(real_panel):
    """The shift-direction test, on the data that actually produces the report."""
    correlations = _close_target_correlations(real_panel)
    offenders = {k: v for k, v in correlations.items() if v > MAX_CLOSE_TARGET_CORR}
    assert not offenders, (
        f"real panel: close correlates with its own target: {offenders}"
    )
