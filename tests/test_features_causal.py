"""Per-family feature contracts.

Three things are checked here. That every family hands back exactly the index
it was given, because the whole panel is held together by positional alignment
on a (asset, date) MultiIndex and a family that reindexes silently misaligns
every row downstream. That no feature is a restatement of the answer. And that
the registry drops the warmup it says it drops, no more and no less.

Defaults run on the synthetic panel. The correlation checks also run on the
real panel when it is present, because a leak can hide in a family that only
becomes interesting on real data, for instance cross-asset lead-lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import schema
from src.data.targets import TARGET_NAMES, build_targets
from src.features import auxiliary as auxiliary_family
from src.features.registry import FAMILY_MODULES, FAMILY_PREFIXES, build_features
from tests.conftest import requires_real_panel

# A feature that reproduces the target to three nines is the target, arriving
# through a different column name. Genuine features on this data top out
# around |r| = 0.07 against the direction target, so there is a vast empty gap
# between honest and leaking and the threshold does not need to be delicate.
MAX_TARGET_CORRELATION = 0.999

# Same threshold against the contemporaneous close. Features here are returns,
# ratios and cyclic calendar encodings, none of which should reproduce a price
# level. A feature that did would be a proxy for the date, which lets a model
# separate the training era from the test era instead of forecasting.
MAX_CLOSE_CORRELATION = 0.999


def _correlations(frame: pd.DataFrame, series: pd.Series, label: str) -> pd.Series:
    """Absolute correlation of every column with ``series``, largest first.

    Fails loudly if any column comes back NaN. A constant or all-NaN feature
    has an undefined correlation, and dropping it quietly would remove it from
    the leak screen without anyone noticing that it was never checked.
    """
    correlations = frame.corrwith(series).abs()
    undefined = list(correlations.index[correlations.isna()])
    assert not undefined, (
        f"{label}: correlation is undefined for {len(undefined)} feature "
        f"columns, so they are not being screened for leakage at all: "
        f"{undefined}. A column is usually constant or entirely NaN when this "
        "happens."
    )
    return correlations.sort_values(ascending=False)


def _top_correlations(frame: pd.DataFrame, series: pd.Series, label: str = "panel") -> pd.Series:
    return _correlations(frame, series, label).head(10)


# ---------------------------------------------------------------------------
# Index preservation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(FAMILY_MODULES))
def test_family_build_returns_the_index_it_was_given(family, synthetic_panel):
    """build() must return the caller's index, unchanged and in the same order.

    Catches the classic pandas trap where ``groupby(...).rolling(...)`` prepends
    the group key to the index. The result looks fine in isolation but joins
    back onto the panel misaligned, which shifts every feature relative to its
    target by an amount that varies per asset.
    """
    built = FAMILY_MODULES[family].build(synthetic_panel)
    assert not built.empty, f"family {family!r} produced no columns on the synthetic panel"
    assert built.index.equals(synthetic_panel.index), (
        f"family {family!r} returned a different index than it was given: "
        f"{len(built)} rows vs {len(synthetic_panel)}, names "
        f"{list(built.index.names)} vs {list(synthetic_panel.index.names)}"
    )
    assert list(built.index.names) == schema.INDEX_NAMES


def test_auxiliary_family_preserves_the_index_even_with_no_sources(synthetic_panel):
    """The auxiliary family degrades to an empty frame, not to a broken index.

    It takes a second argument and is allowed to return no columns when nothing
    was fetched, so it cannot go through the loop above. It must still carry
    the panel index so the registry can concatenate it.
    """
    built = auxiliary_family.build(synthetic_panel, None)
    assert built.index.equals(synthetic_panel.index), (
        "auxiliary.build returned a frame that is not on the panel index"
    )


def test_registry_output_index_is_a_subset_of_the_panel_in_original_order(
    synthetic_panel, base_config
):
    """X, targets and meta must all share one index, drawn from the panel."""
    matrix = build_features(synthetic_panel, base_config)
    assert matrix.X.index.equals(matrix.targets.index), "X and targets are misaligned"
    assert matrix.X.index.equals(matrix.meta.index), "X and meta are misaligned"
    assert matrix.X.index.isin(synthetic_panel.index).all(), (
        "build_features invented rows that are not in the panel"
    )
    assert matrix.X.index.is_monotonic_increasing, (
        "the feature matrix index is not sorted; sequence models slice "
        "contiguous per-asset history from it and would read shuffled history"
    )


# ---------------------------------------------------------------------------
# Nothing is a restatement of the answer
# ---------------------------------------------------------------------------


def _assert_no_feature_reproduces_a_target(matrix, label: str) -> None:
    for target in TARGET_NAMES:
        if target not in matrix.targets.columns:
            continue
        top = _top_correlations(matrix.X, matrix.targets[target], f"{label} / {target}")
        if top.empty:
            continue
        assert float(top.iloc[0]) <= MAX_TARGET_CORRELATION, (
            f"{label}: feature {top.index[0]!r} has |corr| = {float(top.iloc[0]):.6f} "
            f"with the future target {target!r}, above {MAX_TARGET_CORRELATION}. "
            f"It is the answer under another name.\nTop 10 by |corr|:\n"
            + top.to_string()
        )


def test_no_feature_reproduces_the_future_target(synthetic_panel, base_config):
    """No column may be near-perfectly correlated with a t+1 target.

    The failure message prints the top 10 so the offender is named rather than
    the test simply going red.
    """
    matrix = build_features(synthetic_panel, base_config)
    _assert_no_feature_reproduces_a_target(matrix, "synthetic panel")


def test_the_target_correlation_check_would_catch_a_planted_leak(
    synthetic_panel, base_config
):
    """Positive control for the check above.

    Plants the target into the feature matrix under an innocuous name and
    confirms it is detected. Without this, a correlation routine that silently
    returned NaN for every column would make the real test pass.
    """
    matrix = build_features(synthetic_panel, base_config)
    target = base_config["targets"]["primary"]

    planted = matrix.X.copy()
    planted["px_totally_innocent"] = matrix.targets[target].to_numpy()

    top = _top_correlations(planted, matrix.targets[target], "planted leak control")
    assert top.index[0] == "px_totally_innocent", (
        "the planted leak did not come top of the correlation ranking; the "
        "detection routine is not working"
    )
    assert float(top.iloc[0]) > MAX_TARGET_CORRELATION, (
        f"a column that literally is the target only scored "
        f"{float(top.iloc[0]):.6f}, so the {MAX_TARGET_CORRELATION} threshold "
        "would not catch a real leak"
    )


def test_no_feature_reproduces_the_contemporaneous_close(synthetic_panel, base_config):
    """No feature may be a stand-in for the price level.

    Every family here emits returns, ratios, z-scores or cyclic calendar
    encodings, all of which are scale free by construction. A column tracking
    close would encode the date, and on an expanding walk-forward split that
    alone separates train from test.
    """
    matrix = build_features(synthetic_panel, base_config)
    top = _top_correlations(matrix.X, matrix.meta[schema.CLOSE], "synthetic panel / close")
    assert float(top.iloc[0]) <= MAX_CLOSE_CORRELATION, (
        f"feature {top.index[0]!r} has |corr| = {float(top.iloc[0]):.6f} with "
        f"close at the same timestamp, above {MAX_CLOSE_CORRELATION}. No "
        "feature in this codebase is documented as a price level.\n"
        "Top 10 by |corr|:\n" + top.to_string()
    )


# ---------------------------------------------------------------------------
# Warmup accounting
# ---------------------------------------------------------------------------


def _assert_warmup_dropped_exactly(panel, config, label: str) -> None:
    warmup = int(config["features"]["max_lookback"])
    matrix = build_features(panel, config)

    position = panel.groupby(level=schema.ASSET, observed=True).cumcount()
    assets = list(panel.index.get_level_values(schema.ASSET).unique())

    for asset in assets:
        kept = matrix.X.xs(asset, level=schema.ASSET, drop_level=False)
        assert len(kept) > 0, f"{label}: no rows survived for {asset}"
        first_kept = kept.index[0]
        got = int(position.loc[first_kept])
        assert got == warmup, (
            f"{label}: the first surviving {asset} row sits at position {got} "
            f"of the original panel, expected exactly {warmup}. Dropping fewer "
            "means partially filled long windows reached the model; dropping "
            "more silently discards usable history."
        )

    # The row count has to reconcile too. The registry keeps a row when it is
    # past the warmup AND its primary target is not NaN, so the expected count
    # is recomputed from those two conditions rather than assumed to be
    # "warmup + 1 per asset". Assuming the latter would hide an interior NaN
    # target: a row silently dropped from the middle of an asset's history.
    primary = config["targets"]["primary"]
    target_series = build_targets(panel)[primary]
    expected_keep = int(((position >= warmup) & target_series.notna()).sum())
    assert len(matrix.X) == expected_keep, (
        f"{label}: build_features kept {len(matrix.X)} rows, expected "
        f"{expected_keep} (position >= {warmup} and a non-NaN {primary})"
    )

    # And separately, state the assumption rather than leaving it implicit:
    # the only NaN primary targets should be the final bar of each asset.
    nan_rows = target_series.index[target_series.isna()]
    final_bars = {
        (asset, panel.xs(asset, level=schema.ASSET).index.max()) for asset in assets
    }
    interior = [key for key in nan_rows if key not in final_bars]
    assert not interior, (
        f"{label}: {len(interior)} rows have a NaN {primary} away from the end "
        f"of an asset's history, first few {interior[:5]}. Those rows are being "
        "dropped from training without anyone accounting for them."
    )
    assert len(panel) - len(matrix.X) == len(assets) * (warmup + 1), (
        f"{label}: dropped {len(panel) - len(matrix.X)} rows, expected "
        f"{len(assets)} assets x ({warmup} warmup + 1 final bar)"
    )


def test_registry_drops_exactly_the_warmup_it_claims(synthetic_panel, base_config):
    """First retained row per asset is at position max_lookback of the panel."""
    _assert_warmup_dropped_exactly(synthetic_panel, base_config, "synthetic panel")


@pytest.mark.parametrize("warmup", [30, 63, 90])
def test_warmup_accounting_holds_at_several_lookbacks(
    synthetic_panel, base_config, warmup
):
    """The same accounting must hold whatever max_lookback is configured.

    Catches a hardcoded warmup constant that happens to agree with base.yaml.
    """
    config = base_config
    config["features"]["max_lookback"] = warmup
    _assert_warmup_dropped_exactly(synthetic_panel, config, f"warmup={warmup}")


# ---------------------------------------------------------------------------
# Family prefixes
# ---------------------------------------------------------------------------


def test_every_feature_column_carries_a_known_family_prefix(
    synthetic_panel, base_config
):
    """family_of must never return "unknown".

    The registry has no separate column-to-family map: the prefix is the map.
    A column with an unrecognised prefix cannot be switched off in the ablation
    study, so it would quietly stay in every run including the ones that claim
    to have removed it.
    """
    matrix = build_features(synthetic_panel, base_config)
    unknown = [c for c in matrix.X.columns if matrix.family_of(c) == "unknown"]
    assert not unknown, (
        f"feature columns with no known family prefix: {unknown}. "
        f"Known prefixes: {sorted(FAMILY_PREFIXES.values())}"
    )


def test_each_family_only_emits_columns_with_its_own_prefix(synthetic_panel):
    """A family's columns must all start with that family's prefix.

    Catches a column that carries a valid prefix but the wrong one, which would
    make the ablation attribute it to the wrong family.
    """
    for family, module in sorted(FAMILY_MODULES.items()):
        prefix = FAMILY_PREFIXES[family]
        built = module.build(synthetic_panel)
        strays = [c for c in built.columns if not c.startswith(prefix)]
        assert not strays, (
            f"family {family!r} emitted columns not starting with {prefix!r}: {strays}"
        )


def test_family_prefixes_are_mutually_unambiguous():
    """No prefix may be a prefix of another, or family_of becomes order dependent.

    ``family_of`` returns the first prefix that matches while iterating a dict.
    If one prefix were a prefix of another, the answer would depend on the
    insertion order of FAMILY_PREFIXES rather than on the column name.
    """
    prefixes = sorted(FAMILY_PREFIXES.values())
    for i, left in enumerate(prefixes):
        for right in prefixes[i + 1 :]:
            assert not right.startswith(left) and not left.startswith(right), (
                f"family prefixes {left!r} and {right!r} are ambiguous"
            )


# ---------------------------------------------------------------------------
# Real panel checks. Skipped when data/interim/panel_1d.parquet is absent.
# ---------------------------------------------------------------------------


@requires_real_panel
@pytest.mark.parametrize("family", sorted(FAMILY_MODULES))
def test_real_panel_family_build_returns_the_index_it_was_given(family, real_panel):
    """Index preservation on the real panel, where assets start on different dates."""
    built = FAMILY_MODULES[family].build(real_panel)
    assert built.index.equals(real_panel.index), (
        f"real panel: family {family!r} returned a different index than it was given"
    )


@requires_real_panel
def test_real_panel_no_feature_reproduces_the_future_target(real_panel, base_config):
    """The correlation screen on the data that actually produces the report."""
    matrix = build_features(real_panel, base_config)
    _assert_no_feature_reproduces_a_target(matrix, "real panel")


@requires_real_panel
def test_real_panel_no_feature_reproduces_the_contemporaneous_close(
    real_panel, base_config
):
    """No feature on the real panel may be a proxy for the price level."""
    matrix = build_features(real_panel, base_config)
    top = _top_correlations(matrix.X, matrix.meta[schema.CLOSE], "real panel / close")
    assert float(top.iloc[0]) <= MAX_CLOSE_CORRELATION, (
        f"real panel: feature {top.index[0]!r} has |corr| = "
        f"{float(top.iloc[0]):.6f} with the contemporaneous close.\n"
        "Top 10 by |corr|:\n" + top.to_string()
    )


@requires_real_panel
def test_real_panel_registry_drops_exactly_the_warmup_it_claims(
    real_panel, base_config
):
    """Warmup accounting on assets with genuinely different listing dates."""
    _assert_warmup_dropped_exactly(real_panel, base_config, "real panel")


@requires_real_panel
def test_real_panel_every_feature_column_carries_a_known_family_prefix(
    real_panel, base_config
):
    """family_of must never return "unknown" on the real feature matrix."""
    matrix = build_features(real_panel, base_config)
    unknown = [c for c in matrix.X.columns if matrix.family_of(c) == "unknown"]
    assert not unknown, f"real panel: columns with no known family prefix: {unknown}"


@requires_real_panel
def test_real_panel_targets_have_no_infinite_values(real_panel):
    """A target of +/- inf would silently poison a metric.

    log of a zero Parkinson variance is the way this happens; the target code
    masks non-positive variances for exactly this reason.
    """
    targets = build_targets(real_panel)
    for column in targets.columns:
        values = targets[column].to_numpy(dtype="float64")
        assert not np.isinf(values).any(), (
            f"real panel: target {column!r} contains infinite values"
        )
