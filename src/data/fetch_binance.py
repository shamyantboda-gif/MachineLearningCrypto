"""Downloader for the Binance public data archive.

data.binance.vision is a static file host: one zipped CSV of kline bars per
symbol per interval per month, each with a companion .CHECKSUM file. No key,
no account, no rate limit worth working around.

Two details in here are not guessable and both come from reading real files:

1. Timestamp units changed partway through the archive's life. Older months
   are milliseconds since epoch, newer months are microseconds. Assuming
   either one gives you dates in 1970 or dates in the year 55000, so
   :func:`_to_utc` detects the unit from the magnitude of the first value.
2. Some months ship a header row and some do not, so the first line is
   sniffed rather than assumed.

A 404 is not an error here. Months before a symbol was listed and months not
yet published both return 404, which is why the caller gets a per-month log
back alongside the data.
"""

from __future__ import annotations

import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src import schema
from src.config import RAW_DIR

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
USER_AGENT = "crypto-forecasting-research/1.0"

# Epoch magnitudes for any date between roughly 2001 and 2286, used to tell
# milliseconds and microseconds apart without trusting the file's age.
_MS_MIN, _MS_MAX = 1_000_000_000_000, 10_000_000_000_000
_US_MIN, _US_MAX = 1_000_000_000_000_000, 10_000_000_000_000_000


@dataclass
class MonthResult:
    """Outcome of one month's download, kept for the data quality report."""

    month: str
    status: str  # "ok" | "unavailable" | "checksum_failed"
    rows: int = 0


def month_range(start_month: str, end_month: str) -> list[str]:
    """Inclusive list of ``YYYY-MM`` strings between the two endpoints."""
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _url(symbol: str, interval: str, month: str) -> str:
    return f"{BASE_URL}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"


def _download(url: str, timeout: int = 60, retries: int = 3) -> bytes | None:
    """Return the response body, or None for a 404.

    A 404 means "this month does not exist", which is expected. Anything else
    is retried a few times before it is allowed to fail the run, because a
    transient network blip should not cost a 100 file download.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            last_error = error
        except Exception as error:
            last_error = error
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def _verify_checksum(payload: bytes, checksum_body: bytes) -> bool:
    """The .CHECKSUM file holds ``<sha256>  <filename>`` on one line."""
    expected = checksum_body.decode("utf-8", "replace").split()[0].strip().lower()
    return hashlib.sha256(payload).hexdigest() == expected


def _to_utc(values: pd.Series) -> pd.Series:
    """Convert epoch integers to tz-aware UTC, detecting the unit per value.

    The unit has to be decided row by row, not once for the series. Binance
    switched the kline archive from milliseconds to microseconds partway
    through, so a symbol's full history contains both and a single symbol's
    concatenated frame is genuinely mixed. Detecting from the first row alone
    reads every later bar as a date around the year 56971.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    non_null = numeric.dropna()
    if non_null.empty:
        raise ValueError("no parseable timestamps in this file")

    unrecognised = non_null[(non_null < _MS_MIN) | (non_null >= _US_MAX)]
    if not unrecognised.empty:
        raise ValueError(
            f"timestamp magnitude {unrecognised.iloc[0]:.0f} is neither "
            "milliseconds nor microseconds"
        )

    # Normalise everything to microseconds first, then convert once.
    is_microseconds = numeric >= _US_MIN
    micros = numeric.where(is_microseconds, numeric * 1000.0)

    # Pin the resolution. pandas 3.0 infers ms or us from the input, and a
    # panel built from a mix of resolutions fails to align on the date index.
    return pd.to_datetime(micros, unit="us", utc=True).astype("datetime64[ns, UTC]")


def _parse_zip(payload: bytes) -> pd.DataFrame:
    """Unzip one monthly file and return its rows as strings."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        name = archive.namelist()[0]
        with archive.open(name) as handle:
            raw = handle.read()

    text = raw.decode("utf-8", "replace")
    first_line = text.split("\n", 1)[0]
    has_header = first_line.lower().lstrip().startswith("open_time")

    frame = pd.read_csv(
        io.StringIO(text),
        header=0 if has_header else None,
        names=schema.KLINE_COLUMNS,
        dtype=str,
        skiprows=1 if has_header else 0,
    )
    return frame


def _tidy(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Reduce raw kline columns to the subset the panel keeps, typed and in UTC."""
    out = pd.DataFrame()
    out[schema.DATE] = _to_utc(raw["open_time"])
    for column in schema.OHLC:
        out[column] = pd.to_numeric(raw[column], errors="coerce")
    out[schema.VOLUME] = pd.to_numeric(raw["volume"], errors="coerce")
    out[schema.QUOTE_VOLUME] = pd.to_numeric(raw["quote_asset_volume"], errors="coerce")
    out[schema.TRADES] = (
        pd.to_numeric(raw["number_of_trades"], errors="coerce").fillna(0).astype("int64")
    )
    out[schema.TAKER_BUY_QUOTE_VOLUME] = pd.to_numeric(
        raw["taker_buy_quote_volume"], errors="coerce"
    )
    out["symbol"] = symbol

    # A bar with no close price is not a bar. Dropping is correct here and is
    # not the same as forward filling, which the project forbids.
    out = out.dropna(subset=[schema.DATE, schema.CLOSE])
    out = out.drop_duplicates(subset=[schema.DATE], keep="last")
    return out.sort_values(schema.DATE).reset_index(drop=True)


def fetch_symbol(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    verify_checksums: bool = True,
    cache_dir: Path | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, list[MonthResult]]:
    """Download every available month for one symbol and cache the result.

    The cache is the freeze point required by the project's data hygiene rules:
    once written, a run reads the same bytes every time until someone passes
    ``force=True``.
    """
    cache_dir = cache_dir or (RAW_DIR / "binance")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"

    if cache_path.exists() and not force:
        cached = pd.read_parquet(cache_path)
        return cached, [MonthResult("cached", "ok", len(cached))]

    chunks: list[pd.DataFrame] = []
    log: list[MonthResult] = []

    for month in month_range(start_month, end_month):
        url = _url(symbol, interval, month)
        payload = _download(url)
        if payload is None:
            log.append(MonthResult(month, "unavailable"))
            continue

        if verify_checksums:
            checksum_body = _download(url + ".CHECKSUM")
            if checksum_body is not None and not _verify_checksum(payload, checksum_body):
                log.append(MonthResult(month, "checksum_failed"))
                raise RuntimeError(
                    f"sha256 mismatch for {symbol} {month}, refusing to use the file"
                )

        frame = _parse_zip(payload)
        chunks.append(frame)
        log.append(MonthResult(month, "ok", len(frame)))

    if not chunks:
        raise RuntimeError(
            f"no data returned for {symbol} between {start_month} and {end_month}"
        )

    tidy = _tidy(pd.concat(chunks, ignore_index=True), symbol)
    tidy.to_parquet(cache_path, index=False)
    return tidy, log


def fetch_all(
    symbols: dict[str, str],
    interval: str,
    start_month: str,
    end_month: str,
    verify_checksums: bool = True,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[MonthResult]]]:
    """Fetch every configured symbol. Returns asset code to frame, plus the logs."""
    frames: dict[str, pd.DataFrame] = {}
    logs: dict[str, list[MonthResult]] = {}

    for symbol, asset in symbols.items():
        frame, log = fetch_symbol(
            symbol,
            interval,
            start_month,
            end_month,
            verify_checksums=verify_checksums,
            force=force,
        )
        frames[asset] = frame
        logs[asset] = log

        months_ok = sum(1 for entry in log if entry.status == "ok")
        months_missing = sum(1 for entry in log if entry.status == "unavailable")
        first = frame[schema.DATE].min().date()
        last = frame[schema.DATE].max().date()
        detail = (
            "from cache"
            if any(entry.month == "cached" for entry in log)
            else f"{months_ok} months fetched, {months_missing} unavailable"
        )
        print(f"  {symbol:<9s} {len(frame):>5d} bars  {first} .. {last}  ({detail})")

    return frames, logs
