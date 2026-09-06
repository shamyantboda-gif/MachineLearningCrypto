"""Ingest validation.

These checks run on every panel build. Some of them fail the run and some of
them only record a finding, and the split is deliberate: a negative price is a
bug, whereas a 50 percent daily move is a Tuesday in crypto and only needs to
be traceable to a real event.

Everything collected here ends up in ``reports/data_quality.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import schema


@dataclass
class Finding:
    check: str
    severity: str  # "error" | "warning" | "info"
    message: str
    detail: pd.DataFrame | None = None

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.message}"


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)

    def add(
        self,
        check: str,
        severity: str,
        message: str,
        detail: pd.DataFrame | None = None,
    ) -> None:
        self.findings.append(Finding(check, severity, message, detail))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "\n".join(f"  {f}" for f in self.errors)
            raise ValueError(f"panel validation failed:\n{joined}")


def _by_asset(frame: pd.DataFrame):
    return frame.groupby(level=schema.ASSET, observed=True)


def validate_panel(panel: pd.DataFrame, outlier_threshold: float = 0.5) -> ValidationReport:
    """Run every structural check against the assembled panel."""
    report = ValidationReport()
    schema.check_panel_index(panel)

    _check_timezone(panel, report)
    _check_monotonic(panel, report)
    _check_gaps(panel, report)
    _check_ohlc(panel, report)
    _check_volume(panel, report)
    _check_return_outliers(panel, report, outlier_threshold)
    _check_coverage(panel, report)

    return report


def _check_timezone(panel: pd.DataFrame, report: ValidationReport) -> None:
    dates = panel.index.get_level_values(schema.DATE)
    if getattr(dates, "tz", None) is None:
        report.add("timezone", "error", "date index is tz-naive, expected UTC")
        return
    if str(dates.tz) != "UTC":
        report.add("timezone", "error", f"date index is {dates.tz}, expected UTC")
        return
    non_midnight = int(((dates.hour != 0) | (dates.minute != 0)).sum())
    if non_midnight:
        report.add(
            "timezone",
            "error",
            f"{non_midnight} bars do not open at 00:00 UTC",
        )
    else:
        report.add("timezone", "info", "all timestamps are tz-aware UTC at midnight")


def _check_monotonic(panel: pd.DataFrame, report: ValidationReport) -> None:
    problems = []
    for asset, group in _by_asset(panel):
        dates = group.index.get_level_values(schema.DATE)
        if not dates.is_monotonic_increasing:
            problems.append((asset, "not sorted"))
        if dates.has_duplicates:
            problems.append((asset, f"{int(dates.duplicated().sum())} duplicate dates"))
    if problems:
        report.add(
            "monotonic_timestamps",
            "error",
            "; ".join(f"{a}: {m}" for a, m in problems),
        )
    else:
        report.add(
            "monotonic_timestamps",
            "info",
            "timestamps strictly increasing with no duplicates in every asset",
        )


def _check_gaps(panel: pd.DataFrame, report: ValidationReport) -> None:
    """Crypto trades every day, so a missing calendar day is a real gap."""
    rows = []
    for asset, group in _by_asset(panel):
        dates = pd.DatetimeIndex(group.index.get_level_values(schema.DATE))
        expected = pd.date_range(dates.min(), dates.max(), freq="D", tz="UTC")
        missing = expected.difference(dates)
        for day in missing:
            rows.append({"asset": asset, "missing_date": day})

    if rows:
        detail = pd.DataFrame(rows)
        counts = detail.groupby("asset").size().to_dict()
        report.add(
            "calendar_gaps",
            "warning",
            f"{len(detail)} missing daily bars: {counts}. Left as gaps, not filled.",
            detail,
        )
    else:
        report.add("calendar_gaps", "info", "no missing daily bars in any asset")


def _check_ohlc(panel: pd.DataFrame, report: ValidationReport) -> None:
    high, low = panel[schema.HIGH], panel[schema.LOW]
    open_, close = panel[schema.OPEN], panel[schema.CLOSE]

    non_positive = (panel[schema.OHLC] <= 0).any(axis=1)
    if non_positive.any():
        report.add(
            "ohlc_positive",
            "error",
            f"{int(non_positive.sum())} bars have a non-positive OHLC value",
            panel.loc[non_positive, schema.OHLC].head(20),
        )

    violations = (low > high) | (open_ > high) | (open_ < low) | (close > high) | (close < low)
    if violations.any():
        report.add(
            "ohlc_ordering",
            "error",
            f"{int(violations.sum())} bars violate low <= open,close <= high",
            panel.loc[violations, schema.OHLC].head(20),
        )

    if not non_positive.any() and not violations.any():
        report.add("ohlc_sanity", "info", f"all {len(panel)} bars pass OHLC ordering and positivity")


def _check_volume(panel: pd.DataFrame, report: ValidationReport) -> None:
    volume = panel[schema.VOLUME]
    negative = volume < 0
    if negative.any():
        report.add("volume_sign", "error", f"{int(negative.sum())} bars have negative volume")

    # A zero-volume bar is not a real trading day. Common in thin early history.
    zero = volume == 0
    if zero.any():
        detail = panel.loc[zero, [schema.CLOSE, schema.VOLUME]].reset_index()
        counts = detail.groupby(schema.ASSET, observed=True).size().to_dict()
        report.add(
            "zero_volume",
            "warning",
            f"{int(zero.sum())} zero-volume bars: {counts}",
            detail.head(20),
        )
    else:
        report.add("zero_volume", "info", "no zero-volume bars")


def _check_return_outliers(
    panel: pd.DataFrame, report: ValidationReport, threshold: float
) -> None:
    log_close = np.log(panel[schema.CLOSE])
    returns = _by_asset(log_close.to_frame("v"))["v"].diff()
    extreme = returns.abs() > threshold

    if extreme.any():
        detail = (
            pd.DataFrame({"log_return": returns[extreme]})
            .reset_index()
            .sort_values("log_return", key=abs, ascending=False)
        )
        report.add(
            "return_outliers",
            "warning",
            f"{int(extreme.sum())} daily log returns exceed {threshold:.0%} in absolute value. "
            "Each should trace to a real event rather than a data error.",
            detail.head(25),
        )
    else:
        report.add("return_outliers", "info", f"no daily log return exceeds {threshold:.0%}")


def _check_coverage(panel: pd.DataFrame, report: ValidationReport) -> None:
    rows = []
    for asset, group in _by_asset(panel):
        dates = group.index.get_level_values(schema.DATE)
        rows.append(
            {
                "asset": asset,
                "rows": len(group),
                "first": dates.min().date(),
                "last": dates.max().date(),
                "years": round((dates.max() - dates.min()).days / 365.25, 2),
            }
        )
    report.add(
        "coverage",
        "info",
        "per-asset coverage",
        pd.DataFrame(rows),
    )


def cross_source_agreement(
    binance: pd.DataFrame,
    coinbase: pd.DataFrame,
    max_abs_pct_diff: float = 1.0,
    min_agreement_frac: float = 0.99,
) -> tuple[Finding, pd.DataFrame]:
    """Compare daily closes from two exchanges on the same UTC dates.

    Binance quotes USDT and Coinbase quotes USD, so a small persistent basis is
    expected and is not an error. Days where the gap blows out are usually an
    exchange outage or a flash crash on one venue, and those are worth naming.
    """
    left = binance[[schema.DATE, schema.CLOSE]].rename(columns={schema.CLOSE: "binance"})
    right = coinbase[[schema.DATE, schema.CLOSE]].rename(columns={schema.CLOSE: "coinbase"})
    merged = left.merge(right, on=schema.DATE, how="inner")

    if merged.empty:
        return (
            Finding(
                "cross_source",
                "warning",
                "no overlapping dates between the two sources, check skipped",
            ),
            merged,
        )

    merged["pct_diff"] = 100.0 * (merged["binance"] - merged["coinbase"]) / merged["coinbase"]
    within = merged["pct_diff"].abs() <= max_abs_pct_diff
    frac = float(within.mean())

    # A single pooled agreement number hides where the disagreement lives, and
    # in practice it is not spread evenly. Breaking it out by year is what makes
    # the check actionable rather than just alarming.
    by_year = (
        merged.assign(year=merged[schema.DATE].dt.year, ok=within)
        .groupby("year")
        .agg(
            days=("ok", "size"),
            agree_frac=("ok", "mean"),
            median_basis_pct=("pct_diff", "median"),
            worst_abs_pct=("pct_diff", lambda s: s.abs().max()),
        )
        .round(4)
        .reset_index()
    )

    severity = "info" if frac >= min_agreement_frac else "warning"
    message = (
        f"{frac:.2%} of {len(merged)} overlapping days agree within {max_abs_pct_diff:.2f}%. "
        f"Median basis {merged['pct_diff'].median():+.3f}%, "
        f"worst {merged['pct_diff'].abs().max():.2f}%. "
        "See the per-year table below before reading the pooled figure: Binance quotes USDT "
        "and Coinbase quotes USD, and the two venues only converge once cross-exchange "
        "arbitrage in this pair became routine."
    )
    return Finding("cross_source", severity, message, by_year), merged
