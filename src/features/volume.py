"""Volume, trade count and order flow features, prefix ``vol_``.

Raw volume levels are not comparable across assets or across years, so almost
everything here is either a log level or a ratio against the asset's own recent
history. ``log1p`` is used in place of ``log`` throughout because a zero volume
bar is a real thing in this data and ``log(0)`` would poison the column with
``-inf`` rather than an honest NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

QUOTE_VOLUME_Z_WINDOW = 21
QUOTE_VOLUME_MEDIAN_WINDOW = 63
TRADES_Z_WINDOW = 21
TAKER_MEAN_WINDOW = 7
RETURN_VOLUME_CORR_WINDOW = 21


def _by_asset(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` within each asset and return a series on the panel index."""
    return series.groupby(level=schema.ASSET, observed=True).transform(fn)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide and turn any non-finite result into NaN.

    Quote volume, trade count and rolling dispersion can all legitimately be
    zero on a dead bar. Dividing by that yields inf or a signed inf that would
    survive winsorisation and dominate whatever scaler runs downstream, so the
    result is demoted to NaN and dropped with the rest of the warmup.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.where(np.isfinite(result))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score of ``series`` against its own trailing window, per asset."""
    mean = _by_asset(series, lambda values: values.rolling(window).mean())
    std = _by_asset(series, lambda values: values.rolling(window).std())
    return _safe_ratio(series - mean, std)


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the ``vol_`` feature block for ``panel``, on the panel's own index."""
    schema.check_panel_index(panel)

    volume = panel[schema.VOLUME]
    quote_volume = panel[schema.QUOTE_VOLUME]
    trades = panel[schema.TRADES].astype("float64")
    taker_buy = panel[schema.TAKER_BUY_QUOTE_VOLUME]

    log_volume = np.log1p(volume)
    log_quote_volume = np.log1p(quote_volume)
    log_trades = np.log1p(trades)

    features: dict[str, pd.Series] = {
        "vol_log1p_volume": log_volume,
        "vol_log1p_quote_volume": log_quote_volume,
    }

    features[f"vol_zscore_{QUOTE_VOLUME_Z_WINDOW}"] = _rolling_zscore(
        log_quote_volume, QUOTE_VOLUME_Z_WINDOW
    )

    # Median rather than mean for the longer window: volume distributions have a
    # heavy right tail and a single exchange listing event would drag a mean
    # denominator around for a full quarter.
    quote_volume_median = _by_asset(
        quote_volume,
        lambda values: values.rolling(QUOTE_VOLUME_MEDIAN_WINDOW).median(),
    )
    features[f"vol_qv_median_ratio_{QUOTE_VOLUME_MEDIAN_WINDOW}"] = _safe_ratio(
        quote_volume, quote_volume_median
    )

    features[f"vol_trades_zscore_{TRADES_Z_WINDOW}"] = _rolling_zscore(
        log_trades, TRADES_Z_WINDOW
    )

    # Share of quote volume that came from aggressive buyers. A bar with zero
    # quote volume has no defined buy share, so the guard sends it to NaN.
    taker_ratio = _safe_ratio(taker_buy, quote_volume)
    features["vol_taker_buy_ratio"] = taker_ratio
    features[f"vol_taker_buy_ratio_mean_{TAKER_MEAN_WINDOW}"] = _by_asset(
        taker_ratio, lambda values: values.rolling(TAKER_MEAN_WINDOW).mean()
    )
    # Balanced flow sits at 0.5, so the centred version is the signed imbalance
    # and is the one a linear model can use without an intercept per asset.
    features["vol_taker_buy_ratio_dev"] = taker_ratio - 0.5

    # Whether volume is currently confirming or fading price moves. The 1-day
    # return is recomputed here rather than imported from the price family so
    # that every build() depends only on the raw OHLCV columns.
    log_close = np.log(panel[schema.CLOSE])
    ret_1 = log_close - _by_asset(log_close, lambda values: values.shift(1))
    pair = pd.concat([ret_1.rename("ret"), log_quote_volume.rename("logqv")], axis=1)
    # A rolling correlation takes two series, which ``transform`` cannot express,
    # and ``groupby().apply()`` collapses to a wide frame when the panel holds a
    # single asset. Iterating the groups keeps one code path for both cases and
    # makes the per-asset scoping impossible to misread.
    blocks = [
        block["ret"].rolling(RETURN_VOLUME_CORR_WINDOW).corr(block["logqv"])
        for _, block in pair.groupby(level=schema.ASSET, observed=True)
    ]
    features[f"vol_ret_corr_{RETURN_VOLUME_CORR_WINDOW}"] = pd.concat(blocks).reindex(
        panel.index
    )

    return pd.DataFrame(features, index=panel.index)
