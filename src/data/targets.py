"""Target construction, defined in exactly one place.

Every target here answers a question about ``t+1`` using a row indexed at
``t``. The row indexed ``t`` therefore holds features from ``t`` and a target
from ``t+1``, and nothing downstream is allowed to shift either one again.
That single convention is what ``tests/test_alignment.py`` checks.

Returns are modelled, never price levels. A model trained on levels reaches an
R squared near 0.99 by learning to repeat yesterday's close, which looks
excellent and forecasts nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

RET = "ret_1d"
DIR = "dir_1d"
VOL = "vol_1d"

# Columns the models must never see as features but the evaluator needs.
FWD_SIMPLE_RETURN = "fwd_ret_simple"
REALISED_VAR = "realised_var_1d"

TARGET_NAMES = [RET, DIR, VOL]

# Parkinson's estimator scales the squared log range by this constant to make
# it an unbiased estimate of variance under a driftless diffusion.
_PARKINSON_SCALE = 1.0 / (4.0 * np.log(2.0))


def _by_asset(frame: pd.DataFrame):
    return frame.groupby(level=schema.ASSET, observed=True)


def parkinson_variance(high: pd.Series, low: pd.Series) -> pd.Series:
    """Single bar variance estimate from the high-low range.

    With only daily bars there are no intraday returns to sum, so the range
    stands in for realised variance. It uses strictly more information than a
    close-to-close squared return and is the standard choice at this frequency.
    """
    log_range = np.log(high / low)
    return _PARKINSON_SCALE * log_range.pow(2)


def build_targets(panel: pd.DataFrame) -> pd.DataFrame:
    """Return every target and evaluation column, indexed like ``panel``.

    ``panel`` must carry the (asset, date) MultiIndex. All shifts are negative
    because a target is a fact about the future; every feature elsewhere in the
    codebase uses non-negative shifts only.
    """
    schema.check_panel_index(panel)

    close = panel[schema.CLOSE]
    log_close = np.log(close)

    out = pd.DataFrame(index=panel.index)

    # Target A: next-day log return.
    next_log_close = _by_asset(log_close.to_frame("v"))["v"].shift(-1)
    out[RET] = next_log_close - log_close

    # Target B: next-day direction. Left as NaN wherever the return is NaN so
    # the final bar of each asset drops out rather than becoming a fake zero.
    out[DIR] = np.where(out[RET].notna(), (out[RET] > 0).astype("float64"), np.nan)

    # Target C: log of next-day Parkinson variance. Modelled in logs because
    # variance is strictly positive and heavily right skewed.
    realised = parkinson_variance(panel[schema.HIGH], panel[schema.LOW])
    realised = realised.where(realised > 0)
    next_realised = _by_asset(realised.to_frame("v"))["v"].shift(-1)
    out[REALISED_VAR] = next_realised
    out[VOL] = np.log(next_realised)

    # Simple return over t to t+1, which is what a position held at t earns.
    # Log returns do not compound arithmetically across a portfolio, so the
    # backtest needs this and not Target A.
    next_close = _by_asset(close.to_frame("v"))["v"].shift(-1)
    out[FWD_SIMPLE_RETURN] = next_close / close - 1.0

    return out


def target_column(name: str) -> str:
    """Validate a target name coming from config."""
    if name not in TARGET_NAMES:
        raise ValueError(f"unknown target {name!r}, expected one of {TARGET_NAMES}")
    return name


def describe_class_balance(targets: pd.DataFrame) -> pd.DataFrame:
    """Base rate of the direction target per asset.

    Reported per fold as well. If a classifier's accuracy equals the base rate
    it has learned to always predict up, which is not a forecast.
    """
    frame = targets[[DIR]].dropna()
    grouped = frame.groupby(level=schema.ASSET, observed=True)[DIR]
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "base_rate_up": grouped.mean(),
        }
    )
