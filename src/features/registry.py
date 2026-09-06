"""Assembles the feature matrix from whichever families the config enables.

Toggling families here is what makes the ablation study possible: turn one off,
rerun, compare. Because each family owns a column prefix, the registry can
always say which family a column came from without keeping a separate map.

Feature *computation* happens globally on the frozen panel and that is safe: a
rolling mean ending at ``t`` uses only data at or before ``t``, so computing it
once for the whole history gives the same answer as computing it fold by fold.
Feature *scaling* is a different matter and lives in
:mod:`src.preprocess`, inside the fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import schema
from src.config import PROCESSED_DIR, config_hash
from src.data.targets import build_targets
from src.features import auxiliary as auxiliary_family
from src.features import calendar_feats, cross_asset, price, range_vol, technical, volume

# Config family name to the module that builds it. The keys match the
# `features.families` block of config/base.yaml.
FAMILY_MODULES = {
    "price": price,
    "volume": volume,
    "range": range_vol,
    "technical": technical,
    "cross_asset": cross_asset,
    "calendar": calendar_feats,
}

FAMILY_PREFIXES = {
    "price": "px_",
    "volume": "vol_",
    "range": "rng_",
    "technical": "tech_",
    "cross_asset": "xa_",
    "calendar": "cal_",
    "auxiliary": "aux_",
}


@dataclass
class FeatureMatrix:
    """Features, targets and the columns the evaluator needs but models must not see."""

    X: pd.DataFrame
    targets: pd.DataFrame
    meta: pd.DataFrame
    families: dict[str, list[str]]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    def family_of(self, column: str) -> str:
        for family, prefix in FAMILY_PREFIXES.items():
            if column.startswith(prefix):
                return family
        return "unknown"

    def summary(self) -> pd.DataFrame:
        rows = [
            {"family": family, "n_features": len(columns)}
            for family, columns in sorted(self.families.items())
        ]
        rows.append({"family": "TOTAL", "n_features": self.X.shape[1]})
        return pd.DataFrame(rows)


def build_features(
    panel: pd.DataFrame,
    config: dict,
    auxiliary: dict[str, pd.DataFrame] | None = None,
) -> FeatureMatrix:
    """Build every enabled family, join them, and drop the warmup period."""
    schema.check_panel_index(panel)

    enabled = config["features"]["families"]
    parts: list[pd.DataFrame] = []
    families: dict[str, list[str]] = {}

    for name, module in FAMILY_MODULES.items():
        if not enabled.get(name, False):
            continue
        built = module.build(panel)
        if built.empty:
            continue
        _assert_index_preserved(name, panel, built)
        parts.append(built)
        families[name] = list(built.columns)

    if enabled.get("auxiliary", False):
        built = auxiliary_family.build(panel, auxiliary)
        if not built.empty:
            _assert_index_preserved("auxiliary", panel, built)
            parts.append(built)
            families["auxiliary"] = list(built.columns)

    if not parts:
        raise ValueError("no feature families are enabled, nothing to build")

    X = pd.concat(parts, axis=1)
    duplicates = X.columns[X.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"two families produced the same column name: {duplicates}")

    targets = build_targets(panel)
    meta = _build_meta(panel, targets)

    X, targets, meta = _drop_warmup(X, targets, meta, config)
    return FeatureMatrix(X=X, targets=targets, meta=meta, families=families)


def _build_meta(panel: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Columns the evaluator and the simpler models need, kept out of ``X``.

    The baselines and the classical time series models work directly on the
    return series rather than on engineered features, so a lagged return and
    the current realised variance live here. They are deliberately not in ``X``:
    putting them there would mean the ablation could not turn the price family
    off, and it would let a model reach them by a second route.
    """
    import numpy as np

    from src.data.targets import parkinson_variance

    close = panel[schema.CLOSE]
    log_close = np.log(close)

    meta = panel[[schema.CLOSE, schema.HIGH, schema.LOW]].copy()
    meta["ret_lag1"] = log_close.groupby(level=schema.ASSET, observed=True).diff()
    meta["realised_var_now"] = parkinson_variance(panel[schema.HIGH], panel[schema.LOW])
    return meta.join(targets)


def _assert_index_preserved(name: str, panel: pd.DataFrame, built: pd.DataFrame) -> None:
    if not built.index.equals(panel.index):
        raise ValueError(
            f"feature family {name!r} returned a different index than it was given"
        )


def _drop_warmup(
    X: pd.DataFrame,
    targets: pd.DataFrame,
    meta: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove the leading rows where long windows have not filled yet.

    Two things are dropped and they are separate. The first ``max_lookback``
    rows of each asset are dropped because their long-window features are
    partially undefined, and imputing them would invent history. The final row
    of each asset is dropped because its target refers to a bar that does not
    exist yet.
    """
    warmup = int(config["features"].get("max_lookback", 63))

    position = X.groupby(level=schema.ASSET, observed=True).cumcount()
    keep = position >= warmup

    primary = config["targets"]["primary"]
    keep = keep & targets[primary].notna()

    return X.loc[keep], targets.loc[keep], meta.loc[keep]


def features_path(config: dict) -> str:
    """Cache path for a feature matrix, named by the hash of the settings that made it."""
    relevant = {
        "features": config["features"],
        "data": {k: config["data"][k] for k in ("symbols", "interval", "start_month", "end_month")},
        "targets": config["targets"],
    }
    return str(PROCESSED_DIR / f"features_{config_hash(relevant)}.parquet")
