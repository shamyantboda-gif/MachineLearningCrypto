"""Cross-asset features.

If genuine signal exists anywhere in this dataset it most likely lives here.
Single-asset technical features are deterministic transforms of one price
series, whereas the lead-lag structure between assets is real information that
a per-asset model cannot see at all.

Everything is lagged. The BTC return that appears as a feature for ETH on date
``t`` is the return BTC realised over ``t-1`` to ``t``, which is public before
the ``t`` to ``t+1`` return being predicted. A same-day BTC return would be a
leak, and a subtle one, because BTC and ETH are correlated strongly enough that
it would look like skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

PREFIX = "xa_"
REFERENCE_ASSET = "BTC"

_CORR_WINDOWS = (30, 90)


def build(panel: pd.DataFrame, reference: str = REFERENCE_ASSET) -> pd.DataFrame:
    """Cross-asset features for every row of ``panel``."""
    schema.check_panel_index(panel)

    log_close = np.log(panel[schema.CLOSE])
    returns = log_close.groupby(level=schema.ASSET, observed=True).diff()

    # Wide view, one column per asset, so cross-sectional operations are simple.
    wide = returns.unstack(level=schema.ASSET)
    assets = list(wide.columns)

    out = pd.DataFrame(index=panel.index)

    if reference in assets:
        out = _add_reference_features(out, panel, wide, returns, reference)
    else:
        # The panel can legitimately lack the reference asset, for instance in a
        # single-asset ablation. Degrade rather than fail.
        print(f"  cross_asset: reference asset {reference} absent, skipping lead-lag features")

    out = _add_cross_sectional_features(out, panel, wide)
    return out


def _broadcast(series_by_date: pd.Series, index: pd.MultiIndex, name: str) -> pd.Series:
    """Map a date-indexed series onto every (asset, date) row."""
    dates = index.get_level_values(schema.DATE)
    return pd.Series(series_by_date.reindex(dates).to_numpy(), index=index, name=name)


def _add_reference_features(
    out: pd.DataFrame,
    panel: pd.DataFrame,
    wide: pd.DataFrame,
    returns: pd.Series,
    reference: str,
) -> pd.DataFrame:
    """Features describing each asset's relationship to the reference asset."""
    reference_returns = wide[reference]

    # BTC leads the market, so yesterday's BTC move is a candidate predictor for
    # everything including BTC itself. The shift is what keeps it causal.
    for lag in (1, 2):
        out[f"{PREFIX}ref_ret_lag{lag}"] = _broadcast(
            reference_returns.shift(lag), panel.index, "ref"
        )

    out[f"{PREFIX}ref_ret_mean_5"] = _broadcast(
        reference_returns.shift(1).rolling(5).mean(), panel.index, "ref"
    )
    out[f"{PREFIX}ref_vol_21"] = _broadcast(
        reference_returns.shift(1).rolling(21).std(), panel.index, "ref"
    )

    # Rolling correlation with the reference. Computed on the wide frame so each
    # asset keeps its own history, then stacked back to long format.
    for window in _CORR_WINDOWS:
        corr_wide = wide.rolling(window).corr(reference_returns)
        corr_long = corr_wide.stack(future_stack=True).reorder_levels(schema.INDEX_NAMES)
        out[f"{PREFIX}corr_ref_{window}"] = corr_long.reindex(panel.index)

    # Relative strength: how far the asset's own trailing move sits from the
    # reference's over the same window.
    for window in (7, 30):
        own = returns.groupby(level=schema.ASSET, observed=True).rolling(window).sum()
        own = own.reset_index(level=0, drop=True).reindex(panel.index)
        ref_cum = _broadcast(reference_returns.rolling(window).sum(), panel.index, "ref")
        out[f"{PREFIX}rel_strength_{window}"] = own - ref_cum

    return out


def _add_cross_sectional_features(
    out: pd.DataFrame, panel: pd.DataFrame, wide: pd.DataFrame
) -> pd.DataFrame:
    """Where each asset sits within the panel on a given date."""
    n_assets = wide.shape[1]

    # Percentile rank of yesterday's return across the panel. Rank is used
    # rather than the raw value because it is invariant to a market-wide move,
    # which isolates the idiosyncratic part.
    lagged = wide.shift(1)
    ranks = lagged.rank(axis=1, pct=True)
    out[f"{PREFIX}rank_ret_lag1"] = (
        ranks.stack(future_stack=True).reorder_levels(schema.INDEX_NAMES).reindex(panel.index)
    )

    # Equal-weight panel return, lagged. A crude market factor.
    market = lagged.mean(axis=1)
    out[f"{PREFIX}market_ret_lag1"] = _broadcast(market, panel.index, "mkt")
    out[f"{PREFIX}market_ret_mean_5"] = _broadcast(market.rolling(5).mean(), panel.index, "mkt")

    # Cross-sectional dispersion. Wide dispersion means the assets are moving on
    # their own news, narrow dispersion means one factor is driving everything.
    if n_assets > 1:
        out[f"{PREFIX}market_dispersion"] = _broadcast(
            lagged.std(axis=1), panel.index, "disp"
        )

    # Excess over the market, which is the part a cross-sectional model can act on.
    out[f"{PREFIX}excess_ret_lag1"] = (
        lagged.sub(market, axis=0)
        .stack(future_stack=True)
        .reorder_levels(schema.INDEX_NAMES)
        .reindex(panel.index)
    )

    return out
