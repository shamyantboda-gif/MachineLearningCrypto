"""Hand-rolled technical indicators, prefix ``tech_``.

An honest note on what this family can and cannot contribute. RSI, MACD,
Bollinger position and the stochastic oscillator are all deterministic functions
of the same lagged close series that ``price.py`` already exposes directly. They
carry no information the price block does not, and a sufficiently flexible model
fed the raw return lags could in principle learn any of them. What they do add
is a specific nonlinearity for free: bounded oscillators, ratios against a
rolling band, and exponential rather than rectangular weighting. That shortens
the path a tree or a linear model has to travel to a useful decision boundary.
Expect them to help through inductive bias, not through new information, and do
not read a high importance score on them as evidence of a novel signal.

No indicator library is used. ``ta`` and ``pandas-ta`` are not dependencies of
this project and the formulas below are short enough that a hand-rolled version
is easier to audit than a wrapper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STDS = 2.0
STOCHASTIC_WINDOW = 14
STOCHASTIC_SMOOTH = 3
OBV_Z_WINDOW = 21


def _by_asset(series: pd.Series, fn) -> pd.Series:
    """Apply ``fn`` within each asset and return a series on the panel index."""
    return series.groupby(level=schema.ASSET, observed=True).transform(fn)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide and turn any non-finite result into NaN.

    Every denominator in this module can hit zero on a flat stretch: a bar with
    no gains and no losses, a Bollinger band of zero width, a 14 day window with
    no high-low separation. Those are undefined indicator readings rather than
    infinite ones, so they become NaN and are dropped with the warmup.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    return result.where(np.isfinite(result))


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with the classic recursive weighting."""
    return _by_asset(series, lambda values: values.ewm(span=span, adjust=False).mean())


def _wilder(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing, which is an EMA with ``alpha = 1 / window``."""
    return _by_asset(
        series, lambda values: values.ewm(alpha=1.0 / window, adjust=False).mean()
    )


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the ``tech_`` feature block for ``panel``, on the panel's own index."""
    schema.check_panel_index(panel)

    high = panel[schema.HIGH]
    low = panel[schema.LOW]
    close = panel[schema.CLOSE]
    volume = panel[schema.VOLUME]

    features: dict[str, pd.Series] = {}

    delta = close - _by_asset(close, lambda values: values.shift(1))
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    average_gain = _wilder(gain, RSI_WINDOW)
    average_loss = _wilder(loss, RSI_WINDOW)
    # Algebraically identical to 100 - 100 / (1 + gain / loss) but it never
    # forms the intermediate infinity when the window contains no losses.
    features[f"tech_rsi_{RSI_WINDOW}"] = 100.0 * _safe_ratio(
        average_gain, average_gain + average_loss
    )

    macd_line = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    macd_signal = _ema(macd_line, MACD_SIGNAL)
    # The raw MACD line is a difference of two prices, so a 2017 bitcoin reading
    # and a 2024 one are not on the same scale, and neither is bitcoin against
    # litecoin. Dividing by close converts it into a fraction of price, which is
    # comparable across assets and across price eras. The histogram is taken as
    # the difference of the two normalised series so all three share one unit.
    normalised_line = _safe_ratio(macd_line, close)
    normalised_signal = _safe_ratio(macd_signal, close)
    features["tech_macd_line_norm"] = normalised_line
    features["tech_macd_signal_norm"] = normalised_signal
    features["tech_macd_hist_norm"] = normalised_line - normalised_signal

    middle = _by_asset(close, lambda values: values.rolling(BOLLINGER_WINDOW).mean())
    band_std = _by_asset(close, lambda values: values.rolling(BOLLINGER_WINDOW).std())
    lower = middle - BOLLINGER_STDS * band_std
    width = 2.0 * BOLLINGER_STDS * band_std
    features[f"tech_bb_pos_{BOLLINGER_WINDOW}"] = _safe_ratio(close - lower, width)

    lowest_low = _by_asset(low, lambda values: values.rolling(STOCHASTIC_WINDOW).min())
    highest_high = _by_asset(
        high, lambda values: values.rolling(STOCHASTIC_WINDOW).max()
    )
    percent_k = 100.0 * _safe_ratio(close - lowest_low, highest_high - lowest_low)
    features[f"tech_stoch_k_{STOCHASTIC_WINDOW}"] = percent_k
    features[f"tech_stoch_d_{STOCHASTIC_WINDOW}"] = _by_asset(
        percent_k, lambda values: values.rolling(STOCHASTIC_SMOOTH).mean()
    )

    signed_volume = np.sign(delta) * volume
    on_balance_volume = _by_asset(signed_volume, lambda values: values.cumsum())
    obv_mean = _by_asset(
        on_balance_volume, lambda values: values.rolling(OBV_Z_WINDOW).mean()
    )
    obv_std = _by_asset(
        on_balance_volume, lambda values: values.rolling(OBV_Z_WINDOW).std()
    )
    # The raw OBV level is a cumulative sum, so it grows without bound and its
    # scale depends on how long the asset has been listed and how large its
    # volume happens to be. Handing that to a model trained on early folds and
    # tested on late ones leaks a monotone proxy for the date. The 21 day
    # z-score keeps the shape of the accumulation and discards the level.
    features[f"tech_obv_zscore_{OBV_Z_WINDOW}"] = _safe_ratio(
        on_balance_volume - obv_mean, obv_std
    )

    return pd.DataFrame(features, index=panel.index)
