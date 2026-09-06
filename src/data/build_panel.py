"""Assemble the canonical panel from raw downloads.

``data/interim/panel_1d.parquet`` is the single source of truth for everything
downstream. It holds OHLCV only. Auxiliary series live beside it in
``auxiliary.parquet`` rather than inside it, because they arrive on different
calendars and mixing them in would hide that fact.

Run as a module:

    python -m src.data.build_panel --config config/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import schema
from src.config import INTERIM_DIR, REPORTS_DIR, ensure_dirs, load_config
from src.data import validate as validation
from src.data.fetch_binance import fetch_all

PANEL_PATH = INTERIM_DIR / "panel_1d.parquet"
AUXILIARY_PATH = INTERIM_DIR / "auxiliary.parquet"
DATA_QUALITY_PATH = REPORTS_DIR / "data_quality.md"


def assemble(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack per-asset frames into the long panel."""
    parts = []
    for asset, frame in frames.items():
        part = frame.copy()
        part[schema.ASSET] = asset
        part[schema.SOURCE] = "binance"
        parts.append(part[schema.PANEL_COLUMNS])

    panel = pd.concat(parts, ignore_index=True)
    panel[schema.ASSET] = panel[schema.ASSET].astype("category")
    panel[schema.SOURCE] = panel[schema.SOURCE].astype("category")
    for column, dtype in schema.PANEL_DTYPES.items():
        panel[column] = panel[column].astype(dtype)
    return schema.set_panel_index(panel)


def load_panel(path: Path | None = None) -> pd.DataFrame:
    """Read the frozen panel back, indexed and sorted."""
    path = path or PANEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m src.data.build_panel` to rebuild it."
        )
    return schema.set_panel_index(pd.read_parquet(path))


def load_auxiliary(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read whichever auxiliary series were successfully fetched."""
    path = path or AUXILIARY_PATH
    if not path.exists():
        return {}
    stored = pd.read_parquet(path)
    return {name: group.drop(columns="_source") for name, group in stored.groupby("_source")}


def _collect_auxiliary(config: dict) -> pd.DataFrame | None:
    """Fetch auxiliary series, tolerating every one of them being unavailable."""
    try:
        from src.data.fetch_auxiliary import fetch_auxiliary_all
    except ImportError:
        print("  auxiliary fetchers not available, skipping")
        return None

    sources = fetch_auxiliary_all(config)
    parts = []
    for name, frame in sources.items():
        if frame is None or frame.empty:
            continue
        tagged = frame.copy()
        tagged["_source"] = name
        parts.append(tagged)

    if not parts:
        return None
    return pd.concat(parts, ignore_index=True)


def _cross_check(config: dict, panel: pd.DataFrame) -> tuple[validation.Finding | None, pd.DataFrame]:
    settings = config["data"].get("cross_check", {})
    if not settings.get("enabled", False):
        return None, pd.DataFrame()

    try:
        from src.data.fetch_coinbase import fetch_coinbase_daily
    except ImportError:
        print("  coinbase fetcher not available, cross-check skipped")
        return None, pd.DataFrame()

    asset = settings.get("asset", "BTC")
    dates = panel.xs(asset, level=schema.ASSET).index
    start, end = str(dates.min().date()), str(dates.max().date())

    try:
        coinbase = fetch_coinbase_daily(f"{asset}-USD", start, end)
    except Exception as error:
        print(f"  coinbase cross-check unavailable: {error}")
        return None, pd.DataFrame()

    if coinbase is None or coinbase.empty:
        print("  coinbase returned no rows, cross-check skipped")
        return None, pd.DataFrame()

    binance = panel.xs(asset, level=schema.ASSET).reset_index()
    finding, merged = validation.cross_source_agreement(
        binance,
        coinbase,
        max_abs_pct_diff=settings.get("max_abs_pct_diff", 1.0),
        min_agreement_frac=settings.get("min_agreement_frac", 0.99),
    )
    return finding, merged


def _format_detail(detail: pd.DataFrame | None, limit: int = 15) -> str:
    if detail is None or detail.empty:
        return ""
    shown = detail.head(limit)
    body = shown.to_markdown(index=False)
    suffix = f"\n\n_{len(detail)} rows total, first {limit} shown._" if len(detail) > limit else ""
    return f"\n\n{body}{suffix}\n"


def write_data_quality_report(
    panel: pd.DataFrame,
    report: validation.ValidationReport,
    cross_finding: validation.Finding | None,
    auxiliary: pd.DataFrame | None,
    path: Path = DATA_QUALITY_PATH,
) -> None:
    """Write ``reports/data_quality.md``.

    This file is regenerated on every build and is deliberately not committed.
    """
    dates = panel.index.get_level_values(schema.DATE)
    lines: list[str] = [
        "# Data quality report",
        "",
        f"Generated from `{PANEL_PATH.name}` on {pd.Timestamp.utcnow():%Y-%m-%d %H:%M} UTC.",
        "",
        f"- Rows: {len(panel):,}",
        f"- Assets: {', '.join(sorted(panel.index.get_level_values(schema.ASSET).unique()))}",
        f"- Date range: {dates.min().date()} to {dates.max().date()}",
        "",
        "## Checks",
        "",
    ]

    order = {"error": 0, "warning": 1, "info": 2}
    for finding in sorted(report.findings, key=lambda f: order[f.severity]):
        lines.append(f"### {finding.check} ({finding.severity})")
        lines.append("")
        lines.append(finding.message)
        detail = _format_detail(finding.detail)
        if detail:
            lines.append(detail)
        lines.append("")

    lines.append("## Cross-source agreement")
    lines.append("")
    if cross_finding is None:
        lines.append("Not run. The cross-check source was disabled or unreachable.")
    else:
        lines.append(f"**{cross_finding.severity}**: {cross_finding.message}")
        detail = _format_detail(cross_finding.detail)
        if detail:
            lines.append("")
            lines.append("Agreement by year:")
            lines.append(detail)
    lines.append("")

    lines.append("## Auxiliary series")
    lines.append("")
    if auxiliary is None or auxiliary.empty:
        lines.append("None fetched. The auxiliary feature family will be empty for this run.")
    else:
        summary = (
            auxiliary.groupby("_source")
            .agg(rows=("_source", "size"), first=(schema.DATE, "min"), last=(schema.DATE, "max"))
            .reset_index()
        )
        lines.append(summary.to_markdown(index=False))
    lines.append("")

    lines.append("## Handling rules applied")
    lines.append("")
    lines.extend(
        [
            "- Missing bars are left missing. Prices are never forward filled, because a "
            "filled price silently creates a zero return and biases volatility downward.",
            "- Zero-volume bars are flagged rather than dropped, so the choice stays visible.",
            "- Return outliers are flagged rather than winsorised at this stage. Winsorisation "
            "happens inside the training fold, never on the full series.",
            "- All timestamps are UTC. A daily bar opens at 00:00 UTC and a daily return is "
            "close to close in UTC.",
        ]
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {path}")


def build(config: dict, force: bool = False) -> pd.DataFrame:
    """Fetch, assemble, validate and freeze the panel."""
    ensure_dirs()
    data_cfg = config["data"]

    print("Fetching Binance archive")
    frames, _ = fetch_all(
        data_cfg["symbols"],
        data_cfg["interval"],
        data_cfg["start_month"],
        data_cfg["end_month"],
        verify_checksums=data_cfg.get("verify_checksums", True),
        force=force,
    )

    panel = assemble(frames)

    print("Validating panel")
    report = validate_panel_and_print(panel)

    print("Cross-checking against a second exchange")
    cross_finding, _ = _cross_check(config, panel)
    if cross_finding is not None:
        print(f"  {cross_finding}")

    print("Fetching auxiliary series")
    auxiliary = _collect_auxiliary(config)
    if auxiliary is not None:
        auxiliary.to_parquet(AUXILIARY_PATH, index=False)
        print(f"  wrote {AUXILIARY_PATH} ({len(auxiliary):,} rows)")

    panel.reset_index().to_parquet(PANEL_PATH, index=False)
    print(f"  wrote {PANEL_PATH} ({len(panel):,} rows)")

    write_data_quality_report(panel, report, cross_finding, auxiliary)
    report.raise_if_failed()
    return panel


def validate_panel_and_print(panel: pd.DataFrame) -> validation.ValidationReport:
    report = validation.validate_panel(panel)
    for finding in report.findings:
        if finding.severity != "info":
            print(f"  {finding}")
    print(
        f"  {len(report.errors)} errors, {len(report.warnings)} warnings, "
        f"{len(report.findings)} checks run"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical daily panel.")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the raw cache exists",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    build(config, force=args.force)


if __name__ == "__main__":
    main()
