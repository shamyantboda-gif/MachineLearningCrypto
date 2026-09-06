"""Calendar features, prefix ``cal_``.

The only family here that needs no history at all: every value is a function of
the timestamp alone, so these columns are fully populated from the first row of
each asset and contribute nothing to the warmup.

Each periodic field is encoded as a sine and cosine pair rather than as a raw
integer. A raw integer imposes a false ordering, telling the model that Sunday
is six units away from Monday when it is in fact adjacent, and that December is
eleven months from January. The pair places each value on a circle so that the
distance between two dates reflects their real calendar proximity, and both
components stay bounded in [-1, 1] with no scaling required.

Phases are measured from zero: day of week uses 0 for Monday as pandas reports
it, day of month subtracts one so the first of the month sits at phase zero, and
month of year subtracts one so January does. Day of month is given a fixed
period of 31 rather than the true length of each month. A varying period would
make the same phase mean a different thing in February than in March, and the
fixed version keeps the encoding comparable across the year at the cost of a
small discontinuity in short months.

Crypto trades every day, so ``cal_is_weekend`` is not a market closure flag. It
marks the days when traditional venues are shut and participation is thinner,
which is a genuine regime difference in this asset class rather than a gap in
the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import schema

DAYS_IN_WEEK = 7
DAYS_IN_MONTH = 31
MONTHS_IN_YEAR = 12
FIRST_WEEKEND_DAY = 5  # pandas dayofweek: Monday is 0, Saturday is 5.


def _cyclic(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the sine and cosine of a phase already expressed in turns."""
    radians = 2.0 * np.pi * phase
    return np.sin(radians), np.cos(radians)


def build(panel: pd.DataFrame) -> pd.DataFrame:
    """Return the ``cal_`` feature block for ``panel``, on the panel's own index."""
    schema.check_panel_index(panel)

    dates = panel.index.get_level_values(schema.DATE)
    day_of_week = np.asarray(dates.dayofweek, dtype="float64")
    day_of_month = np.asarray(dates.day, dtype="float64")
    month_of_year = np.asarray(dates.month, dtype="float64")

    dow_sin, dow_cos = _cyclic(day_of_week / DAYS_IN_WEEK)
    dom_sin, dom_cos = _cyclic((day_of_month - 1.0) / DAYS_IN_MONTH)
    moy_sin, moy_cos = _cyclic((month_of_year - 1.0) / MONTHS_IN_YEAR)

    features = {
        "cal_dow_sin": dow_sin,
        "cal_dow_cos": dow_cos,
        "cal_dom_sin": dom_sin,
        "cal_dom_cos": dom_cos,
        "cal_moy_sin": moy_sin,
        "cal_moy_cos": moy_cos,
        # Float rather than int so the whole feature matrix stays one dtype and
        # the downstream scaler does not have to special case this column.
        "cal_is_weekend": (day_of_week >= FIRST_WEEKEND_DAY).astype("float64"),
    }

    return pd.DataFrame(features, index=panel.index)
