"""Range based volatility estimators, prefix ``rng_``.

Close-to-close volatility throws away everything that happened inside the bar.
Parkinson and Garman-Klass use the high, low and open as well, which makes them
several times more efficient per observation at the cost of assuming the bar is
a driftless diffusion sampled continuously. The final ratio feature is there to
expose exactly where that assumption breaks: when close-to-close vol runs well
above the range estimate, the day moved mostly through gaps or a trend rather
than through two sided noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

PARKINSON_WINDOWS = (5, 21)
GARMAN_KLASS_WINDOWS = (5, 21)
ATR_WINDOW = 14
VOL_RATIO_WINDOW = 21

_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2.0))
_GARMAN_KLASS_CROSS = 2.0 * np.log(2.0) - 1.0


def _by_asset(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` within each asset and return a series on the panel index."""
    return series.groupby(level=schema.ASSET, observed=True).transform(fn)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide and turn any non-finite result into NaN.

    A bar where high equals low is a real occurrence on illiquid days and on
    exchange outages. The intraday position of the close is undefined there, and
    an inf would be worse than a missing value for every model downstream.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.where(np.isfinite(result))


def _safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Log of a price ratio, NaN wherever the ratio is not strictly positive."""
    ratio = _safe_ratio(numerator, denominator)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log(ratio)
    return result.where(np.isfinite(result))


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the ``rng_`` feature block for ``panel``, on the panel's own index."""
    schema.check_panel_index(panel)

    open_ = panel[schema.OPEN]
    high = panel[schema.HIGH]
    low = panel[schema.LOW]
    close = panel[schema.CLOSE]

    log_hl = _safe_log_ratio(high, low)
    log_co = _safe_log_ratio(close, open_)

    features: dict[str, pd.Series] = {}

    parkinson_term = _PARKINSON_SCALE * log_hl.pow(2)
    parkinson: dict[int, pd.Series] = {}
    for window in PARKINSON_WINDOWS:
        mean_term = _by_asset(
            parkinson_term, lambda values, window=window: values.rolling(window).mean()
        )
        parkinson[window] = np.sqrt(mean_term)
        features[f"rng_parkinson_{window}"] = parkinson[window]

    garman_klass_term = 0.5 * log_hl.pow(2) - _GARMAN_KLASS_CROSS * log_co.pow(2)
    for window in GARMAN_KLASS_WINDOWS:
        mean_term = _by_asset(
            garman_klass_term,
            lambda values, window=window: values.rolling(window).mean(),
        )
        # The cross term can outweigh the range term on a bar that opened and
        # closed at the extremes, so the window mean is occasionally slightly
        # negative. Clipping at zero before the square root keeps the estimator
        # real without inventing a magnitude.
        features[f"rng_garman_klass_{window}"] = np.sqrt(mean_term.clip(lower=0.0))

    prev_close = _by_asset(close, lambda values: values.shift(1))
    true_range = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # ``max(axis=1)`` skips NaN, which would quietly produce a two term true
    # range on the first bar of each asset. The previous close is the only NaN
    # source here, so masking on it keeps the warmup honest.
    true_range = true_range.where(prev_close.notna())

    average_true_range = _by_asset(
        true_range, lambda values: values.rolling(ATR_WINDOW).mean()
    )
    # Both are in price units, which are meaningless across assets and across a
    # decade of price levels. Dividing by close turns them into daily move sizes.
    features["rng_tr_norm"] = _safe_ratio(true_range, close)
    features[f"rng_atr_{ATR_WINDOW}_norm"] = _safe_ratio(average_true_range, close)

    log_close = np.log(close)
    ret_1 = log_close - _by_asset(log_close, lambda values: values.shift(1))
    close_to_close = _by_asset(
        ret_1, lambda values: values.rolling(VOL_RATIO_WINDOW).std()
    )
    features[f"rng_cc_parkinson_ratio_{VOL_RATIO_WINDOW}"] = _safe_ratio(
        close_to_close, parkinson[VOL_RATIO_WINDOW]
    )

    features["rng_intraday"] = log_hl
    features["rng_close_position"] = _safe_ratio(close - low, high - low)

    return pd.DataFrame(features, index=panel.index)
