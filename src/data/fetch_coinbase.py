"""Coinbase Exchange daily candles, used as an independent price cross-check.

Spec section 4.2. Binance is the primary source; this module exists so that
BTC closes can be reconciled against a second venue. The two disagree by a
small persistent basis because Coinbase quotes USD and Binance quotes USDT,
so the reconciliation reports a percentage difference rather than equality.

Two constraints shape the code here:

1. The candles endpoint caps a response at 300 candles regardless of the range
   requested, and it does not tell you that it truncated. Asking for eight
   years in one call silently returns the most recent 300 days. Every range is
   therefore split into windows of at most 300 days and stitched back together.
2. Coinbase returns ``time`` in whole seconds, whereas Binance uses
   milliseconds. Feeding seconds to a millisecond parser yields dates in 1970,
   which is the kind of error that survives until someone plots it.

The public entry point ``fetch_all_coinbase`` never raises. Coinbase is a
nice-to-have validation source, not a dependency of the panel, and a US or
VPN geo-block on the endpoint must not take the pipeline down with it. Only
the private pagination helper raises, so that genuine HTTP failures are still
visible to a caller that wants them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src import schema
from src.config import RAW_DIR

CANDLES_URL = "https://api.exchange.coinbase.com/products/{product_id}/candles"

# 86400 seconds is the only granularity that gives daily bars.
GRANULARITY_DAILY = 86400

# Hard server-side cap on candles per response.
MAX_CANDLES_PER_REQUEST = 300

# Public rate limit is roughly 10 requests/second per IP, shared across all
# callers behind that IP. 0.35s leaves plenty of headroom for a batch job.
REQUEST_SPACING_S = 0.35

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5
REQUEST_TIMEOUT_S = 30

# 429 is the rate limiter and 5xx are Coinbase-side blips. Both clear on their
# own. Anything else (404 for an unknown product, 400 for a bad range) will
# return the same answer however many times it is asked.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Response rows arrive as positional arrays, not objects.
CANDLE_FIELDS = [
    schema.DATE,
    schema.LOW,
    schema.HIGH,
    schema.OPEN,
    schema.CLOSE,
    schema.VOLUME,
]

OUTPUT_COLUMNS = [
    schema.DATE,
    schema.OPEN,
    schema.HIGH,
    schema.LOW,
    schema.CLOSE,
    schema.VOLUME,
]

DEFAULT_PRODUCTS = ["BTC-USD", "ETH-USD", "LTC-USD", "SOL-USD"]

_HEADERS = {"User-Agent": "crypto-price-prediction-research/1.0"}


def _empty_frame() -> pd.DataFrame:
    """An empty result that still carries the dtypes downstream merges expect.

    A bare ``pd.DataFrame(columns=...)`` gives an object-dtype date column,
    which fails to align against the tz-aware dates in the panel.
    """
    frame = pd.DataFrame({schema.DATE: pd.Series([], dtype="datetime64[ns, UTC]")})
    for column in OUTPUT_COLUMNS[1:]:
        frame[column] = pd.Series([], dtype="float64")
    return frame


def _to_iso(moment: pd.Timestamp) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _request_window(
    session: requests.Session,
    product_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> list[list[Any]]:
    """Fetch one window of at most 300 daily candles.

    This is the only function in the module that raises. Callers that need
    graceful degradation catch around it.
    """
    url = CANDLES_URL.format(product_id=product_id)
    params = {
        "granularity": GRANULARITY_DAILY,
        "start": _to_iso(window_start),
        "end": _to_iso(window_end),
    }

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT_S
            )
        except requests.RequestException as error:
            # Connection reset, DNS failure, read timeout. Worth one more try.
            last_error = error
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue

        raise RuntimeError(
            f"coinbase {product_id} {params['start']}..{params['end']} "
            f"returned HTTP {response.status_code}: {response.text[:400]}"
        )

    raise RuntimeError(
        f"coinbase {product_id} {params['start']}..{params['end']} failed after "
        f"{MAX_RETRIES} attempts: {last_error}"
    )


def _paginate(
    session: requests.Session,
    product_id: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[list[Any]]:
    """Walk the range forward in 300 day windows and concatenate the rows.

    Both ``start`` and ``end`` are inclusive on this endpoint, so windows step
    by 300 days and begin the day after the previous one ended. Ranges that
    predate a product's listing return an empty list rather than an error,
    which is why SOL-USD asked for 2017 is not a failure.
    """
    rows: list[list[Any]] = []
    cursor = start
    step = pd.Timedelta(days=MAX_CANDLES_PER_REQUEST - 1)

    while cursor <= end:
        window_end = min(cursor + step, end)
        rows.extend(_request_window(session, product_id, cursor, window_end))
        cursor = window_end + pd.Timedelta(days=1)
        if cursor <= end:
            time.sleep(REQUEST_SPACING_S)

    return rows


def _rows_to_frame(rows: list[list[Any]]) -> pd.DataFrame:
    """Normalise raw candle arrays into the output frame.

    Rows arrive newest-first, and overlapping windows can repeat a boundary
    day, so the frame is sorted and deduped before it is returned.
    """
    if not rows:
        return _empty_frame()

    frame = pd.DataFrame(rows, columns=CANDLE_FIELDS)

    # unit="s" is the whole point: Coinbase epochs are seconds, Binance's are
    # milliseconds, and both are plausible-looking integers.
    frame[schema.DATE] = pd.to_datetime(
        frame[schema.DATE].astype("int64"), unit="s", utc=True
    ).dt.normalize()

    for column in OUTPUT_COLUMNS[1:]:
        frame[column] = frame[column].astype("float64")

    frame = frame[OUTPUT_COLUMNS]
    frame = frame.sort_values(schema.DATE)
    frame = frame.drop_duplicates(subset=schema.DATE, keep="last")
    return frame.reset_index(drop=True)


def fetch_coinbase_daily(
    product_id: str,
    start: str,
    end: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Return daily OHLCV for one Coinbase product.

    Columns are ``date`` (tz-aware UTC midnight), ``open``, ``high``, ``low``,
    ``close`` and ``volume``, sorted ascending with one row per date. Results
    are cached to parquet and reused unless ``force`` is set, because the
    reconciliation report gets re-run far more often than the history changes.
    """
    directory = Path(cache_dir) if cache_dir is not None else RAW_DIR / "coinbase"
    cache_path = directory / f"{product_id}_1d.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()
    if end_ts < start_ts:
        raise ValueError(f"end {end} precedes start {start}")

    with requests.Session() as session:
        rows = _paginate(session, product_id, start_ts, end_ts)

    frame = _rows_to_frame(rows)

    # An empty result is never cached. A range that predates a product's
    # listing legitimately returns nothing, and because the cache key carries
    # no range, persisting that would make every later call for the full
    # history read back zero rows and drop the asset from the cross-check.
    if not frame.empty:
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path, index=False)
    return frame


def fetch_all_coinbase(
    product_ids: list[str],
    start: str,
    end: str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch several products, keyed by asset code (``BTC-USD`` becomes ``BTC``).

    Degrades to an empty dict rather than raising. Coinbase is geo-blocked in
    some jurisdictions and occasionally retires a product id; neither should
    stop the panel from being built, so the caller sees a warning and simply
    skips the cross-check. Products are caught individually so that one bad id
    does not discard the ones that already succeeded.
    """
    results: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    for product_id in product_ids:
        asset = product_id.split("-")[0]
        try:
            frame = fetch_coinbase_daily(product_id, start, end, force=force)
        except Exception as error:
            failures.append(f"{product_id}: {error}")
            continue

        results[asset] = frame
        if frame.empty:
            print(f"[coinbase] {product_id}: 0 rows (no history in range)")
        else:
            first = frame[schema.DATE].iloc[0].date()
            last = frame[schema.DATE].iloc[-1].date()
            print(f"[coinbase] {product_id}: {len(frame)} rows, {first} to {last}")

    if failures:
        print(
            "[coinbase] warning: cross-check source unavailable for "
            f"{len(failures)} product(s); continuing without them."
        )
        for message in failures:
            print(f"[coinbase]   {message}")

    if not results:
        print(
            "[coinbase] warning: no Coinbase data retrieved at all. The price "
            "cross-check in section 4.2 will be skipped."
        )

    return results
