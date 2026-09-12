"""The per-asset backtest must mean "this asset traded alone".

An asset's share of the equal-weight portfolio is a different number, and the
one a reader will not expect. These tests pin the standalone meaning and check
that the per-asset rows and the portfolio row are built from the same inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import cost_sweep
from src.config import load_config
from src.run_backtest import per_asset_sweep, positions_for


def _positions_and_forward(n: int = 300) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    index = pd.MultiIndex.from_product([["BTC", "ETH", "LTC"], dates], names=["asset", "date"])
    positions = pd.Series(rng.choice([-1.0, 0.0, 1.0], size=len(index)), index=index, name="position")
    forward = pd.Series(rng.normal(scale=0.03, size=len(index)), index=index, name="forward_return")
    return positions, forward


def test_per_asset_sweep_equals_running_each_asset_as_its_own_book():
    config = load_config("config/base.yaml")
    positions, forward = _positions_and_forward()

    table = per_asset_sweep(positions, forward, config)

    levels = config["backtest"]["cost_bps_round_trip"]
    assert set(table["asset"]) == {"BTC", "ETH", "LTC"}
    # Every cost level plus buy and hold, per asset.
    assert len(table) == 3 * (len(levels) + 1)

    btc = positions.index.get_level_values("asset") == "BTC"
    alone = cost_sweep(positions[btc], forward[btc], [float(x) for x in levels])
    got = table[table["asset"] == "BTC"].reset_index(drop=True)
    for column in ["annual_return", "sharpe", "max_drawdown", "annual_turnover"]:
        np.testing.assert_allclose(got[column].to_numpy(), alone[column].to_numpy(), equal_nan=True)


def test_per_asset_buy_and_hold_is_that_asset_held():
    config = load_config("config/base.yaml")
    positions, forward = _positions_and_forward()

    table = per_asset_sweep(positions, forward, config)
    bh = table[table["run_type"] == "buy_and_hold"].set_index("asset")

    for asset in ["BTC", "ETH", "LTC"]:
        mask = forward.index.get_level_values("asset") == asset
        expected = float(np.prod(1.0 + forward[mask].to_numpy()) - 1.0)
        assert bh.loc[asset, "total_return"] == pytest.approx(expected, rel=1e-9)


def test_positions_for_averages_seeds_before_applying_the_rule():
    config = load_config("config/base.yaml")
    dates = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC")
    index = pd.MultiIndex.from_product([["BTC"], dates], names=["asset", "date"])
    # Two seeds that disagree: one says 0.60, the other 0.40. Averaged, the
    # signal sits inside the dead zone and the book stays flat.
    frame = pd.concat(
        [
            pd.DataFrame({"pred": 1.0, "proba": 0.60, "model": "m", "seed": 0}, index=index),
            pd.DataFrame({"pred": 0.0, "proba": 0.40, "model": "m", "seed": 1}, index=index),
        ]
    )
    positions = positions_for(frame, "m", config)
    assert (positions == 0.0).all()
