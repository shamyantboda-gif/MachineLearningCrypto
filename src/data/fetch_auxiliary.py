"""Auxiliary feature sources: sentiment, on-chain activity and macro context.

Spec section 4.3. These feed the ``auxiliary`` feature family, which the
ablation study can switch off entirely. That is the design principle for the
whole module: every fetcher here is optional, so every fetcher degrades to an
empty frame with a printed warning instead of raising. A free public API that
is rate limited, geo-blocked or simply down for an afternoon must not be able
to stop a panel rebuild, and a run that quietly lacks the Fear and Greed index
is a far better outcome than a run that does not exist.

The empty frames are constructed with real dtypes rather than
``pd.DataFrame(columns=[...])``. An object-dtype date column will not align
against the tz-aware dates in the panel, so the failure would resurface later
as an empty merge that looks like a data problem rather than an outage.

Three sources, three different timestamp conventions, all normalised to
tz-aware UTC midnight on ingest:
  Fear and Greed  seconds since epoch, delivered as strings
  Coin Metrics    ISO 8601 with nanosecond precision
  FRED            bare calendar dates with no zone at all
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src import schema
from src.config import RAW_DIR, get as config_get

FNG_URL = "https://api.alternative.me/fng/"
COINMETRICS_URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

FNG_COLUMN = "fng"

# Community tier serves these for the major UTXO and account chains.
DEFAULT_COINMETRICS_METRICS = ["AdrActCnt", "TxCnt", "SplyCur"]

DEFAULT_FRED_SERIES = ["DGS10", "VIXCLS"]

MAX_RETRIES = 3
RETRY_BACKOFF_S = 1.5
REQUEST_TIMEOUT_S = 45
REQUEST_SPACING_S = 0.25
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Coin Metrics caps page_size at 10000 rows.
COINMETRICS_PAGE_SIZE = 10000

_HEADERS = {"User-Agent": "crypto-price-prediction-research/1.0"}


class _Forbidden(Exception):
    """Coin Metrics refused an asset/metric pair on the community tier."""


def _empty_frame(float_columns: list[str], with_asset: bool = False) -> pd.DataFrame:
    """Build a correctly typed empty frame for a failed fetch."""
    frame = pd.DataFrame({schema.DATE: pd.Series([], dtype="datetime64[ns, UTC]")})
    if with_asset:
        frame[schema.ASSET] = pd.Series([], dtype="string")
    for column in float_columns:
        frame[column] = pd.Series([], dtype="float64")
    return frame


def _get(session: requests.Session, url: str, params: dict[str, Any] | None) -> requests.Response:
    """GET with bounded retries on transient failures.

    A 403 is raised as ``_Forbidden`` rather than a generic error because Coin
    Metrics uses it to mean "this pair is above your tier", which the caller
    handles by narrowing the request instead of giving up.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                url, params=params, headers=_HEADERS, timeout=REQUEST_TIMEOUT_S
            )
        except requests.RequestException as error:
            last_error = error
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue

        if response.status_code == 200:
            return response
        if response.status_code == 403:
            raise _Forbidden(response.text[:300])
        if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_S * (attempt + 1))
            continue

        raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:300]}")

    raise RuntimeError(f"{url} failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_fear_greed(cache_dir: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Daily Alternative.me Fear and Greed index, 0 to 100, from Feb 2018.

    Returns ``date`` and ``fng``. ``limit=0`` asks for the full history in one
    response, so there is nothing to paginate.
    """
    directory = Path(cache_dir) if cache_dir is not None else RAW_DIR / "auxiliary"
    cache_path = directory / "fear_greed.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    try:
        with requests.Session() as session:
            response = _get(session, FNG_URL, {"limit": 0, "format": "json"})
        records = response.json()["data"]
        frame = pd.DataFrame(records)

        # Both fields arrive as strings, including the epoch.
        frame[schema.DATE] = pd.to_datetime(
            frame["timestamp"].astype("int64"), unit="s", utc=True
        ).dt.normalize()
        frame[FNG_COLUMN] = frame["value"].astype("float64")

        frame = frame[[schema.DATE, FNG_COLUMN]]
        frame = frame.sort_values(schema.DATE).drop_duplicates(
            subset=schema.DATE, keep="last"
        )
        frame = frame.reset_index(drop=True)
    except Exception as error:
        print(f"[auxiliary] warning: Fear and Greed fetch failed ({error}); skipping.")
        return _empty_frame([FNG_COLUMN])

    directory.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def _coinmetrics_pages(
    session: requests.Session,
    assets: str,
    metrics: str,
    start: str,
) -> list[dict[str, Any]]:
    """Follow ``next_page_url`` until it is absent and return every row.

    ``paging_from=start`` is set deliberately. The endpoint defaults to paging
    backwards from the most recent observation, which still terminates but
    makes a truncated run look like recent-only data rather than an obvious
    gap at the front of the history.
    """
    params: dict[str, Any] | None = {
        "assets": assets,
        "metrics": metrics,
        "frequency": "1d",
        "start_time": start,
        "page_size": COINMETRICS_PAGE_SIZE,
        "paging_from": "start",
    }
    url = COINMETRICS_URL
    rows: list[dict[str, Any]] = []

    while url:
        payload = _get(session, url, params).json()
        rows.extend(payload.get("data", []))
        url = payload.get("next_page_url")
        # next_page_url already carries every query parameter plus the token.
        params = None
        if url:
            time.sleep(REQUEST_SPACING_S)

    return rows


def _coinmetrics_frame(rows: list[dict[str, Any]], metrics: list[str]) -> pd.DataFrame:
    """Shape raw Coin Metrics rows into long format with a fixed column set.

    Every requested metric is forced to exist, as all-NaN when the response did
    not carry it, so that the column set of the result does not depend on which
    assets happened to be authorised.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_frame(metrics, with_asset=True)

    frame[schema.DATE] = pd.to_datetime(frame["time"], utc=True, format="mixed").dt.normalize()
    frame[schema.ASSET] = frame["asset"].astype("string").str.upper()

    for metric in metrics:
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce").astype("float64")
        else:
            # float("nan") rather than pd.NA: a NAType scalar cannot be cast
            # into a numpy float64 column under pandas 3.0.
            frame[metric] = float("nan")

    return frame[[schema.DATE, schema.ASSET] + metrics]


def fetch_coinmetrics(
    assets: list[str],
    metrics: list[str] | None = None,
    start: str = "2017-01-01",
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Coin Metrics Community on-chain metrics in long format.

    Returns ``date``, ``asset`` (uppercase) and one float column per metric.

    The community tier does not cover every asset. It rejects an unauthorised
    pair with a 403 that fails the *entire* request, so a single batched call
    for btc,eth,ltc,sol returns nothing at all when only sol is unavailable.
    The request is therefore narrowed on refusal: all assets together first,
    then one asset at a time, then one metric at a time for the asset that
    still refuses. Whatever survives is kept and the rest becomes NaN.
    """
    metric_list = list(metrics) if metrics is not None else list(DEFAULT_COINMETRICS_METRICS)
    lower_assets = [asset.lower() for asset in assets]

    directory = Path(cache_dir) if cache_dir is not None else RAW_DIR / "auxiliary"
    cache_path = directory / "coinmetrics.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    frames: list[pd.DataFrame] = []
    unavailable: list[str] = []

    try:
        with requests.Session() as session:
            try:
                rows = _coinmetrics_pages(
                    session, ",".join(lower_assets), ",".join(metric_list), start
                )
                frames.append(_coinmetrics_frame(rows, metric_list))
            except _Forbidden:
                for asset in lower_assets:
                    try:
                        rows = _coinmetrics_pages(
                            session, asset, ",".join(metric_list), start
                        )
                        frames.append(_coinmetrics_frame(rows, metric_list))
                        continue
                    except _Forbidden:
                        pass

                    # The asset is partially covered at best. Ask per metric so
                    # that one refused metric does not cost the others.
                    for metric in metric_list:
                        try:
                            rows = _coinmetrics_pages(session, asset, metric, start)
                        except _Forbidden:
                            unavailable.append(f"{asset}/{metric}")
                            continue
                        frames.append(_coinmetrics_frame(rows, metric_list))
    except Exception as error:
        print(f"[auxiliary] warning: Coin Metrics fetch failed ({error}); skipping.")
        return _empty_frame(metric_list, with_asset=True)

    if unavailable:
        print(
            "[auxiliary] Coin Metrics community tier does not serve: "
            f"{', '.join(unavailable)}. Those become NaN."
        )

    frames = [candidate for candidate in frames if not candidate.empty]
    if not frames:
        print("[auxiliary] warning: Coin Metrics returned no usable rows; skipping.")
        return _empty_frame(metric_list, with_asset=True)

    # Per-metric fallback produces several partial frames for the same
    # (date, asset). Grouping collapses them onto one row, taking the first
    # non-null value in each metric column, which is an outer join by another
    # name and needs no suffix handling.
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.groupby([schema.DATE, schema.ASSET], as_index=False).first()
    combined = combined.sort_values([schema.ASSET, schema.DATE]).reset_index(drop=True)
    combined = combined[[schema.DATE, schema.ASSET] + metric_list]

    directory.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def _fred_series(session: requests.Session, series_id: str, start: str) -> pd.DataFrame:
    """Download one FRED series as CSV.

    The graph CSV endpoint needs no API key. Its first column has been named
    both ``DATE`` and ``observation_date`` over the years, so it is read by
    position. Missing observations appear either as a literal ``.`` or as an
    empty field depending on the series, and both must become NaN.
    """
    response = _get(session, FRED_CSV_URL, {"id": series_id, "cosd": start})
    frame = pd.read_csv(io.StringIO(response.text), na_values=["."])

    date_column = frame.columns[0]
    frame[schema.DATE] = pd.to_datetime(frame[date_column]).dt.tz_localize("UTC").dt.normalize()
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce").astype("float64")
    return frame[[schema.DATE, series_id]]


def fetch_fred(
    series: list[str],
    start: str = "2017-01-01",
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """FRED macro series in wide format: ``date`` plus one column per series id.

    No forward-filling happens here. FRED publishes on US business days while
    crypto trades every day, so roughly a third of the calendar is genuinely
    absent. Filling it is an assumption about what a weekend yield "is", and
    the panel builder is the one place that makes and documents that choice.
    """
    series_list = list(series)
    directory = Path(cache_dir) if cache_dir is not None else RAW_DIR / "auxiliary"
    cache_path = directory / "fred.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    try:
        frames: list[pd.DataFrame] = []
        with requests.Session() as session:
            for index, series_id in enumerate(series_list):
                if index:
                    time.sleep(REQUEST_SPACING_S)
                frames.append(_fred_series(session, series_id, start))

        combined = frames[0]
        for extra in frames[1:]:
            combined = combined.merge(extra, on=schema.DATE, how="outer")

        combined = combined.sort_values(schema.DATE).drop_duplicates(
            subset=schema.DATE, keep="last"
        )
        combined = combined.reset_index(drop=True)[[schema.DATE] + series_list]
    except Exception as error:
        print(f"[auxiliary] warning: FRED fetch failed ({error}); skipping.")
        return _empty_frame(series_list)

    directory.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    return combined


def fetch_auxiliary_all(config: dict) -> dict[str, pd.DataFrame]:
    """Fetch whichever auxiliary sources ``data.auxiliary`` switches on.

    Keys are ``fear_greed``, ``coinmetrics`` and ``fred``. A disabled source is
    omitted from the result rather than returned empty, so that a caller can
    distinguish "turned off" from "tried and failed".
    """
    toggles = config_get(config, "data.auxiliary", {}) or {}
    results: dict[str, pd.DataFrame] = {}

    if toggles.get("fear_greed"):
        frame = fetch_fear_greed()
        results["fear_greed"] = frame
        print(f"[auxiliary] fear_greed: {_summarise(frame)}")

    if toggles.get("coinmetrics"):
        # Asset codes come from the same symbol map the panel is built from.
        symbols = config_get(config, "data.symbols", {}) or {}
        assets = sorted(set(symbols.values())) or ["BTC"]
        frame = fetch_coinmetrics(assets)
        results["coinmetrics"] = frame
        covered = sorted(frame[schema.ASSET].unique()) if not frame.empty else []
        print(f"[auxiliary] coinmetrics: {_summarise(frame)}, assets {covered}")

    if toggles.get("fred"):
        frame = fetch_fred(DEFAULT_FRED_SERIES)
        results["fred"] = frame
        print(f"[auxiliary] fred: {_summarise(frame)}")

    if not results:
        print("[auxiliary] all auxiliary sources disabled in config.")

    return results


def _summarise(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "0 rows (unavailable)"
    first = frame[schema.DATE].min().date()
    last = frame[schema.DATE].max().date()
    return f"{len(frame)} rows, {first} to {last}"
