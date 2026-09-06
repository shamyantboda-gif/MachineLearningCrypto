"""Price and return features, prefix ``px_``.

Lag convention, stated once so nothing downstream has to guess: ``px_ret_k`` at
time ``t`` is the log return realised over the ``k`` days *ending* at ``t``,
that is ``log(close_t / close_{t-k})``. It is not the 1-day return shifted back
by ``k``. Every lag column in this module follows that rule, so ``px_ret_1`` is
the ordinary 1-day log return and the longer lags are overlapping cumulative
returns rather than isolated past days.

The ``px_cumret_w`` columns measure the same quantity over their own windows but
arrive at it differently: they sum 1-day returns where ``px_ret_k`` differences
two log closes. On a gap-free asset the two agree exactly. They part company
around a missing bar inside the window, which the lag columns bridge over and
the cumulative columns turn into NaN. Both behaviours are wanted, the lags as a
robust momentum reading and the cumulative columns as a strict one.

Everything here is a function of closes at or before ``t``. Rolling windows end
at ``t`` and carry the default ``min_periods == window``, so warmup rows come
out NaN rather than being computed from a short window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

RETURN_LAGS = (1, 2, 3, 5, 10, 21)
MOMENT_WINDOWS = (5, 10, 21, 63)
SHAPE_WINDOWS = (21, 63)
CUMULATIVE_WINDOWS = (7, 30, 90)
EXTREME_WINDOWS = (21, 63)


def _by_asset(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` within each asset and return a series on the panel index.

    ``groupby(...).rolling(...)`` prepends the group key to the index in pandas
    3.x, which breaks the "same index in, same index out" contract. ``transform``
    keeps the original index, and grouping on the asset level guarantees no
    window ever reaches across an asset boundary.
    """
    return series.groupby(level=schema.ASSET, observed=True).transform(fn)


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the ``px_`` feature block for ``panel``, on the panel's own index."""
    schema.check_panel_index(panel)

    log_close = np.log(panel[schema.CLOSE])
    features: dict[str, pd.Series] = {}

    # log(close_t / close_{t-k}) computed as a difference of logs, which is
    # numerically better behaved than a ratio when prices span several orders of
    # magnitude across the panel.
    for lag in RETURN_LAGS:
        features[f"px_ret_{lag}"] = log_close - _by_asset(
            log_close, lambda values, lag=lag: values.shift(lag)
        )

    ret_1 = features["px_ret_1"]

    for window in MOMENT_WINDOWS:
        features[f"px_ret_mean_{window}"] = _by_asset(
            ret_1, lambda values, window=window: values.rolling(window).mean()
        )
        features[f"px_ret_std_{window}"] = _by_asset(
            ret_1, lambda values, window=window: values.rolling(window).std()
        )

    # Third and fourth moments are only meaningful on windows long enough to
    # estimate them, hence 21 and 63 rather than the full moment window set.
    for window in SHAPE_WINDOWS:
        features[f"px_ret_skew_{window}"] = _by_asset(
            ret_1, lambda values, window=window: values.rolling(window).skew()
        )
        features[f"px_ret_kurt_{window}"] = _by_asset(
            ret_1, lambda values, window=window: values.rolling(window).kurt()
        )

    # Summing 1-day log returns rather than differencing log closes so that a
    # single missing bar inside the window invalidates the whole sum instead of
    # silently bridging over it.
    for window in CUMULATIVE_WINDOWS:
        features[f"px_cumret_{window}"] = _by_asset(
            ret_1, lambda values, window=window: values.rolling(window).sum()
        )

    # Distance to the running extremes. Both are logs of a ratio, so they are
    # scale free: the max version is bounded above by 0, the min version below.
    for window in EXTREME_WINDOWS:
        rolling_max = _by_asset(
            log_close, lambda values, window=window: values.rolling(window).max()
        )
        rolling_min = _by_asset(
            log_close, lambda values, window=window: values.rolling(window).min()
        )
        features[f"px_dist_max_{window}"] = log_close - rolling_max
        features[f"px_dist_min_{window}"] = log_close - rolling_min

    return pd.DataFrame(features, index=panel.index)
