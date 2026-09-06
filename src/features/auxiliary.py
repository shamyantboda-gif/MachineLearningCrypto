"""Auxiliary features from sources outside the exchange.

These are the only features that carry information a price series cannot
contain: on-chain activity, a sentiment index, and macro rates. Whether they
help is an empirical question the ablation answers, but including at least one
non-price channel is what stops the feature set from being one variable in
several disguises.

Two alignment problems get handled here and both are stated rather than hidden:

- FRED publishes on US business days and crypto trades every day. Macro values
  are forward filled across weekends. Forward filling is an assumption, not a
  neutral operation, and the alternative of dropping weekends would throw away
  most of the dataset.
- Coin Metrics and the Fear and Greed index publish for a given day at some
  point during or after that day. Every auxiliary series is therefore lagged by
  one day before it becomes a feature, so a row dated ``t`` only ever sees
  values published for ``t-1``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

PREFIX = "aux_"

# Publication lag applied to every auxiliary series, in days.
_PUBLICATION_LAG = 1


def build(panel: pd.DataFrame, auxiliary: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Join the auxiliary sources onto the panel index.

    Returns an empty frame with the right index when no source was fetched, so
    a run with no network access to these APIs still completes.
    """
    schema.check_panel_index(panel)
    out = pd.DataFrame(index=panel.index)

    if not auxiliary:
        return out

    if "fear_greed" in auxiliary:
        out = out.join(_fear_greed_features(panel, auxiliary["fear_greed"]))
    if "coinmetrics" in auxiliary:
        out = out.join(_coinmetrics_features(panel, auxiliary["coinmetrics"]))
    if "fred" in auxiliary:
        out = out.join(_fred_features(panel, auxiliary["fred"]))

    return out


def _daily_index(panel: pd.DataFrame) -> pd.DatetimeIndex:
    dates = panel.index.get_level_values(schema.DATE)
    return pd.DatetimeIndex(sorted(dates.unique()))


def _broadcast(by_date: pd.DataFrame, index: pd.MultiIndex) -> pd.DataFrame:
    """Map a date-indexed frame onto every (asset, date) row of the panel."""
    dates = index.get_level_values(schema.DATE)
    return by_date.reindex(dates).set_axis(index)


def _fear_greed_features(panel: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Level and momentum of the composite sentiment index.

    The index is market wide rather than per asset, so the same value is
    broadcast to every asset on a given date.
    """
    series = (
        source.dropna(subset=[schema.DATE])
        .set_index(schema.DATE)["fng"]
        .sort_index()
        .astype("float64")
    )
    series = series[~series.index.duplicated(keep="last")]
    series = series.reindex(_daily_index(panel)).shift(_PUBLICATION_LAG)

    frame = pd.DataFrame(index=series.index)
    frame[f"{PREFIX}fng"] = series
    frame[f"{PREFIX}fng_chg_7"] = series.diff(7)
    frame[f"{PREFIX}fng_z_63"] = (
        (series - series.rolling(63).mean()) / series.rolling(63).std()
    )
    return _broadcast(frame, panel.index)


def _coinmetrics_features(panel: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """On-chain activity, z-scored per asset.

    Raw levels are non-stationary and grow with adoption, so a model trained on
    them would be reading a time trend. The z-score against a trailing window is
    the part that carries information about the current period.
    """
    metrics = [c for c in source.columns if c not in {schema.DATE, schema.ASSET}]
    if not metrics:
        return pd.DataFrame(index=panel.index)

    tidy = source.dropna(subset=[schema.DATE, schema.ASSET]).copy()
    tidy[schema.ASSET] = tidy[schema.ASSET].astype(str).str.upper()
    tidy = tidy.drop_duplicates(subset=[schema.ASSET, schema.DATE], keep="last")
    tidy = schema.set_panel_index(tidy)

    aligned = tidy.reindex(panel.index)
    grouped = aligned.groupby(level=schema.ASSET, observed=True)

    out = pd.DataFrame(index=panel.index)
    for metric in metrics:
        values = pd.to_numeric(aligned[metric], errors="coerce")
        lagged = grouped[metric].shift(_PUBLICATION_LAG)
        lagged = pd.to_numeric(lagged, errors="coerce")

        # Activity counts are right skewed and strictly positive, so logs first.
        logged = np.log1p(lagged.clip(lower=0))
        by_asset = logged.groupby(level=schema.ASSET, observed=True)
        rolling_mean = by_asset.transform(lambda s: s.rolling(63, min_periods=21).mean())
        rolling_std = by_asset.transform(lambda s: s.rolling(63, min_periods=21).std())

        out[f"{PREFIX}{metric.lower()}_z63"] = (logged - rolling_mean) / rolling_std.replace(0.0, np.nan)
        out[f"{PREFIX}{metric.lower()}_chg_7"] = by_asset.diff(7)

        if values.notna().sum() == 0:
            out = out.drop(
                columns=[f"{PREFIX}{metric.lower()}_z63", f"{PREFIX}{metric.lower()}_chg_7"]
            )

    return out


def _fred_features(panel: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Macro context, forward filled across non-trading days.

    Levels are differenced where a level would be non-stationary. VIX is kept
    as a level because it is mean reverting and its level is the informative
    quantity; the Treasury yield is differenced because its level trends.
    """
    series_columns = [c for c in source.columns if c != schema.DATE]
    if not series_columns:
        return pd.DataFrame(index=panel.index)

    wide = (
        source.dropna(subset=[schema.DATE])
        .set_index(schema.DATE)
        .sort_index()
        .astype("float64")
    )
    wide = wide[~wide.index.duplicated(keep="last")]

    # Reindex onto the crypto calendar, then forward fill the business-day gaps.
    wide = wide.reindex(_daily_index(panel)).ffill().shift(_PUBLICATION_LAG)

    frame = pd.DataFrame(index=wide.index)
    for column in series_columns:
        name = column.lower()
        if column.upper() == "VIXCLS":
            frame[f"{PREFIX}{name}"] = wide[column]
            frame[f"{PREFIX}{name}_chg_5"] = wide[column].diff(5)
        else:
            frame[f"{PREFIX}{name}_chg_1"] = wide[column].diff(1)
            frame[f"{PREFIX}{name}_chg_21"] = wide[column].diff(21)

    return _broadcast(frame, panel.index)
