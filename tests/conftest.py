"""Shared fixtures for the leakage test suite.

Everything here exists so that the three test modules can run fast and without
the data directory. The default panel is synthetic: a realistic multi-asset
OHLCV frame with a (asset, date) MultiIndex, built from a seeded geometric
random walk so the numbers are stable run to run.

The real panel is loaded only if ``data/interim/panel_1d.parquet`` is on disk.
Tests that need it carry the ``requires_real_panel`` marker defined below and
are skipped, not failed, when it is absent.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src import schema
from src.config import INTERIM_DIR, REPO_ROOT, load_config

# ---------------------------------------------------------------------------
# Real panel discovery
# ---------------------------------------------------------------------------

REAL_PANEL_PATH = INTERIM_DIR / "panel_1d.parquet"
REAL_PANEL_AVAILABLE = REAL_PANEL_PATH.exists()

#: Decorate any test that reads the real panel with this.
requires_real_panel = pytest.mark.skipif(
    not REAL_PANEL_AVAILABLE,
    reason=f"real panel not found at {REAL_PANEL_PATH}",
)

BASE_CONFIG_PATH = REPO_ROOT / "config" / "base.yaml"

# Assets used by the synthetic panel. BTC is included deliberately: it is the
# reference asset for the cross_asset family, and without it that family
# silently degrades to cross-sectional features only, which would make the
# cross_asset half of the leakage tests vacuous.
SYNTHETIC_ASSETS = ("BTC", "ETH", "LTC")
SYNTHETIC_START = "2018-01-01"
SYNTHETIC_DAYS = 900


def make_config(**overrides) -> dict:
    """Return a fresh deep copy of the base config.

    A deep copy every time, because several tests flip
    ``features.families.auxiliary`` or ``features.max_lookback``. Sharing one
    dict would let one test's mutation change another test's meaning.
    """
    config = load_config(BASE_CONFIG_PATH)
    # The auxiliary family needs fetched external series that the synthetic
    # panel has no counterpart for. Off by default here; the family itself is
    # still exercised directly in tests/test_features_causal.py.
    config["features"]["families"]["auxiliary"] = False
    for dotted, value in overrides.items():
        node = config
        parts = dotted.split("__")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    return copy.deepcopy(config)


def build_synthetic_panel(
    n_days: int = SYNTHETIC_DAYS,
    assets: tuple[str, ...] = SYNTHETIC_ASSETS,
    start: str = SYNTHETIC_START,
    seed: int = 7,
) -> pd.DataFrame:
    """A seeded multi-asset OHLCV panel with the canonical (asset, date) index.

    Prices follow a geometric random walk with a small positive drift, which is
    what makes ``corr(close[t], close[t+1])`` land near 0.999 the way a real
    price series does. That property is what the shift-direction test in
    tests/test_alignment.py relies on to prove it can detect a wrong-way shift.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=n_days, freq="D", tz="UTC")

    frames = []
    for i, asset in enumerate(assets):
        returns = rng.normal(0.0005, 0.03, n_days)
        close = 100.0 * (1.0 + i) * np.exp(np.cumsum(returns))
        open_ = close * np.exp(rng.normal(0.0, 0.005, n_days))
        high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0.0, 0.01, n_days)))
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0.0, 0.01, n_days)))
        volume = np.exp(rng.normal(10.0, 0.5, n_days))
        # The taker buy share has to vary. A fixed 0.5 would make every
        # vol_taker_buy_ratio column constant, and a constant column has an
        # undefined correlation, which would silently remove it from the
        # feature-versus-target leak screen in test_features_causal.py.
        taker_share = rng.uniform(0.3, 0.7, n_days)
        frames.append(
            pd.DataFrame(
                {
                    schema.ASSET: asset,
                    schema.DATE: dates,
                    schema.OPEN: open_,
                    schema.HIGH: high,
                    schema.LOW: low,
                    schema.CLOSE: close,
                    schema.VOLUME: volume,
                    schema.QUOTE_VOLUME: volume * close,
                    schema.TRADES: rng.integers(1_000, 5_000, n_days),
                    schema.TAKER_BUY_QUOTE_VOLUME: volume * close * taker_share,
                    schema.SOURCE: "synthetic",
                }
            )
        )

    frame = pd.concat(frames, ignore_index=True)
    frame[schema.ASSET] = frame[schema.ASSET].astype("category")
    frame[schema.SOURCE] = frame[schema.SOURCE].astype("category")
    return schema.set_panel_index(frame)


def build_tiny_panel() -> pd.DataFrame:
    """A five-day, two-asset panel with closes chosen by hand.

    The closes are picked so every branch of the direction target is exercised:
    an up day, a down day, a flat day (which must resolve to 0, not 1) and a
    second up day. ETH's price level is an order of magnitude above BTC's so
    that a missing ``groupby(asset)`` in the target code produces a huge, easily
    detected return at the BTC/ETH seam rather than a plausible small one.
    """
    dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")
    closes = {
        "BTC": [100.0, 110.0, 105.0, 105.0, 120.0],
        "ETH": [1000.0, 900.0, 950.0, 1000.0, 1100.0],
    }

    frames = []
    for asset, series in closes.items():
        close = np.asarray(series, dtype="float64")
        frames.append(
            pd.DataFrame(
                {
                    schema.ASSET: asset,
                    schema.DATE: dates,
                    schema.OPEN: close,
                    # High and low are widened symmetrically so the Parkinson
                    # variance target is strictly positive on every bar.
                    schema.HIGH: close * 1.02,
                    schema.LOW: close * 0.98,
                    schema.CLOSE: close,
                    schema.VOLUME: np.full(5, 1_000.0),
                    schema.QUOTE_VOLUME: close * 1_000.0,
                    schema.TRADES: np.full(5, 100, dtype="int64"),
                    schema.TAKER_BUY_QUOTE_VOLUME: close * 500.0,
                    schema.SOURCE: "synthetic",
                }
            )
        )

    # Asset stays a plain string column here, not a category, so that
    # sort_index gives plain lexicographic order (BTC then ETH) and the
    # BTC/ETH seam is at a known position.
    return schema.set_panel_index(pd.concat(frames, ignore_index=True))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_panel_factory():
    """The panel builder itself, for tests that need a custom shape."""
    return build_synthetic_panel


@pytest.fixture(scope="session")
def synthetic_panel() -> pd.DataFrame:
    """The default synthetic panel. Read only; never mutate it in a test."""
    return build_synthetic_panel()


@pytest.fixture(scope="session")
def tiny_panel() -> pd.DataFrame:
    """The hand-computable five-day panel. Read only."""
    return build_tiny_panel()


@pytest.fixture()
def base_config() -> dict:
    """A fresh copy of config/base.yaml with the auxiliary family disabled."""
    return make_config()


@pytest.fixture(scope="session")
def real_panel() -> pd.DataFrame | None:
    """The real panel if it is on disk, otherwise None.

    Returning None rather than raising means a test can depend on this fixture
    and still be skipped cleanly by the requires_real_panel marker.
    """
    if not REAL_PANEL_AVAILABLE:
        return None
    return schema.set_panel_index(pd.read_parquet(REAL_PANEL_PATH))
