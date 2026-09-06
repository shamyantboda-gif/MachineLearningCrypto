"""The canonical panel schema.

``data/interim/panel_1d.parquet`` is the single source of truth for everything
downstream. Long format, one row per asset-date. Every module that touches the
panel imports its column names from here rather than typing string literals.
"""

from __future__ import annotations

import pandas as pd

DATE = "date"
ASSET = "asset"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"
QUOTE_VOLUME = "quote_volume"
TRADES = "trades"
TAKER_BUY_QUOTE_VOLUME = "taker_buy_quote_volume"
SOURCE = "source"

OHLC = [OPEN, HIGH, LOW, CLOSE]

PANEL_COLUMNS = [
    DATE,
    ASSET,
    OPEN,
    HIGH,
    LOW,
    CLOSE,
    VOLUME,
    QUOTE_VOLUME,
    TRADES,
    TAKER_BUY_QUOTE_VOLUME,
    SOURCE,
]

PANEL_DTYPES = {
    OPEN: "float64",
    HIGH: "float64",
    LOW: "float64",
    CLOSE: "float64",
    VOLUME: "float64",
    QUOTE_VOLUME: "float64",
    TRADES: "int64",
    TAKER_BUY_QUOTE_VOLUME: "float64",
}

# Raw Binance kline columns, in file order. The files ship without a header.
KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Feature matrices carry a (asset, date) MultiIndex in this order. Every model
# relies on it: sequence models slice contiguous per-asset history from it.
INDEX_NAMES = [ASSET, DATE]


def set_panel_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` indexed by (asset, date) and sorted."""
    if list(frame.index.names) == INDEX_NAMES:
        return frame.sort_index()
    return frame.set_index(INDEX_NAMES).sort_index()


def check_panel_index(frame: pd.DataFrame) -> None:
    """Raise if ``frame`` is not indexed the way the rest of the code assumes."""
    if list(frame.index.names) != INDEX_NAMES:
        raise ValueError(f"expected index names {INDEX_NAMES}, got {list(frame.index.names)}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted by (asset, date)")
    if frame.index.has_duplicates:
        dupes = frame.index[frame.index.duplicated()][:5].tolist()
        raise ValueError(f"duplicate (asset, date) rows, first few: {dupes}")
