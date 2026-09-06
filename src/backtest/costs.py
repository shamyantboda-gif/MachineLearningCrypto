"""Transaction costs, expressed per unit of turnover.

The only thing this module has to get right is the conversion between a quoted
round trip cost and the cost charged against a single turnover observation.
That conversion is where costless-looking backtests usually come from, so it is
derived explicitly below rather than folded into a magic constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass
class CostModel:
    """Commission and slippage assumptions for one backtest.

    Defaults follow the project spec: Binance spot taker fees are roughly 10 bps
    for a round trip, and 5 bps of slippage per trade is a fair assumption for
    the liquid pairs in this panel.
    """

    commission_bps_round_trip: float = 10.0
    slippage_bps_per_trade: float = 5.0

    @property
    def bps_per_unit_turnover(self) -> float:
        """Basis points charged per 1.0 of turnover.

        Turnover is defined as ``abs(position_t - position_{t-1})``, so a full
        round trip (flat to fully long, then back to flat) is two trades and
        registers turnover of 2.0, not 1.0.

        The requirement is that a round trip costs exactly:

            commission_bps_round_trip + 2 * slippage_bps_per_trade

        Dividing that by the 2.0 of turnover it generates gives the per unit
        rate:

            (commission_bps_round_trip + 2 * slippage_bps_per_trade) / 2
            = commission_bps_round_trip / 2 + slippage_bps_per_trade

        The commission is halved because it was quoted for the round trip; the
        slippage is not, because it was quoted per trade. Charging the full
        commission per unit of turnover would double every commission bill, and
        halving the slippage would let half of it disappear. Both mistakes are
        invisible in the output, which is why the arithmetic is spelled out.
        """
        return 0.5 * self.commission_bps_round_trip + self.slippage_bps_per_trade

    @property
    def rate_per_unit_turnover(self) -> float:
        """Same rate as a decimal fraction, ready to multiply into returns."""
        return self.bps_per_unit_turnover * BPS

    @property
    def round_trip_bps(self) -> float:
        """Total cost, in bps, of one full round trip."""
        return self.commission_bps_round_trip + 2.0 * self.slippage_bps_per_trade

    def cost_for_turnover(self, turnover: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
        """Cost as a decimal fraction of notional, for each turnover observation.

        Returns the same container type it was handed, so a Series keeps its
        (asset, date) index and can be subtracted from a gross return series
        without any realignment.
        """
        if isinstance(turnover, pd.Series):
            return turnover.astype("float64").abs() * self.rate_per_unit_turnover
        return np.abs(np.asarray(turnover, dtype="float64")) * self.rate_per_unit_turnover

    @classmethod
    def from_round_trip_bps(cls, bps: float) -> "CostModel":
        """Build a model whose round trip cost is exactly ``bps``.

        The cost sweep quotes levels such as [0, 5, 20] bps round trip and the
        reported numbers have to be those levels, so the whole charge goes into
        commission and slippage is set to zero. Leaving the default 5 bps of
        slippage in place would silently add 10 bps to every level and the
        sweep's x axis would be a lie.
        """
        return cls(commission_bps_round_trip=float(bps), slippage_bps_per_trade=0.0)

    def __str__(self) -> str:
        return (
            f"CostModel(commission {self.commission_bps_round_trip:g} bps round trip, "
            f"slippage {self.slippage_bps_per_trade:g} bps per trade, "
            f"{self.bps_per_unit_turnover:g} bps per unit turnover, "
            f"{self.round_trip_bps:g} bps per round trip)"
        )
