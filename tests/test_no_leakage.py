"""The project's leakage checklist, turned into executable assertions.

This is the most important file in the repository. A data leak in a project
like this does not crash anything: it produces a directional accuracy of 65
percent, a Sharpe ratio of 3, and a conclusion that is worthless. The only
defence is a test that fails.

Each test below names the specific bug it is looking for. Several of them run a
positive control first, deliberately reproducing the bug and checking that it
would be visible, so that a green tick is evidence rather than decoration.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import schema
from src.config import REPO_ROOT
from src.data.targets import build_targets
from src.features.registry import build_features
from src.models.base import CLASSIFICATION, REGRESSION, FoldData
from src.models.baselines import build_baselines
from src.preprocess import FoldPreprocessor
from src.splits.walk_forward import PurgedWalkForward, splitter_from_config
from tests.conftest import requires_real_panel

SRC_DIR = REPO_ROOT / "src"

# The only module allowed to shift backwards in time. Nothing under
# src/features/ may appear here: a feature that shifts backwards is reaching
# into the future, full stop. Kept as posix relative paths so the comparison
# works the same on Windows and Linux.
FEATURE_NEGATIVE_SHIFT_ALLOWLIST = {"src/data/targets.py"}

# The wider scan over all of src/ carries one extra, deliberately documented
# exception. Every entry needs a reason, and adding one is a decision, not a
# way to get a red test green.
#
# src/models/garch.py reads the GARCH variance recursion with shift(-1). In a
# GARCH(1,1), sigma^2 at t+1 is omega + alpha * eps_t^2 + beta * sigma^2_t, so
# the value the recursion holds for t+1 is already determined by data at or
# before t. Taking the next row is therefore a genuine one step ahead forecast
# rather than a peek, the parameters are frozen with fix() from the training
# fold, and the module says so in its docstring. It is listed here rather than
# left to fail because it is not a leak, but it is the one place in the tree
# where that argument has to be made by hand, so it is written down.
#
# The argument only holds because the recursion is driven directly rather than
# through arch's fix(), which seeds itself from whatever series it is handed and
# would therefore let the test rows influence the backcast and the variance
# bounds. That is what the two GARCH tests further down this file check, and
# they are the reason this entry is a documented exception rather than a general
# licence: the shift is allowed here because it is verified here.
NEGATIVE_SHIFT_ALLOWLIST = FEATURE_NEGATIVE_SHIFT_ALLOWLIST | {"src/models/garch.py"}

# Splitter settings for the 900 day synthetic panel. base.yaml's own numbers
# (first test in 2020, 500 minimum training rows) assume nine years of history
# and would yield zero folds here.
SYNTHETIC_SPLIT = dict(
    train_start="2018-01-01",
    first_test_start="2019-01-01",
    test_months=3,
    step_months=3,
    horizon=1,
    min_train_rows=100,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _scan(pattern: str, root: Path = SRC_DIR) -> list[tuple[str, int, str]]:
    """Return (relative path, 1-based line number, line) for every match."""
    regex = re.compile(pattern)
    hits: list[tuple[str, int, str]] = []
    for path in _python_files(root):
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append((_relative(path), number, line.strip()))
    return hits


def _synthetic_matrix(panel: pd.DataFrame, config: dict):
    return build_features(panel, config)


def _folds(index: pd.MultiIndex, **overrides):
    settings = dict(SYNTHETIC_SPLIT)
    settings.update(overrides)
    settings.setdefault("embargo_days", 5)
    splitter = PurgedWalkForward(**settings)
    return splitter, list(splitter.split(index))


def _dates_of(index: pd.MultiIndex) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(index.get_level_values(schema.DATE))


def _fit_end(index: pd.MultiIndex, fold) -> pd.Timestamp:
    """Latest date of any row the model is allowed to learn from.

    Deliberately the union of train and val, not ``fold.train_end``. The
    validation slice sits between the inner train window and the test window
    and it is used for early stopping, which makes it a fitting row. Measuring
    the purge from ``fold.train_end`` alone would report a gap of two or three
    months on this panel and would still pass with the embargo deleted, which
    is exactly the kind of decorative test this file exists to avoid.
    """
    fitting = (fold.train_mask | fold.val_mask).to_numpy()
    return _dates_of(index)[fitting].max()


# ---------------------------------------------------------------------------
# Static checks on the source tree
# ---------------------------------------------------------------------------


def test_no_centered_rolling_windows_anywhere_in_src():
    """A centered window at t averages over t-k .. t+k, so it sees the future.

    Catches ``rolling(w, center=True)`` and any other centered estimator. This
    is a whole-tree grep rather than a feature-module grep because a centered
    window is wrong in a target, a diagnostic and a plot as well as a feature.
    """
    hits = _scan(r"center\s*=\s*True")
    assert not hits, (
        "centered windows found in src/ (a centered window at t reads data "
        "from after t):\n"
        + "\n".join(f"  {path}:{line}: {text}" for path, line, text in hits)
    )


def test_no_negative_shifts_in_feature_code():
    """No feature module may shift backwards in time. No exceptions here.

    A negative shift is how a target reaches forward to t+1. Inside a feature
    it pulls the future into an input column, which is the leak this whole
    suite exists to prevent. The pattern covers ``.shift(-1)``, ``shift(- 1)``
    and ``shift(-horizon)`` alike.
    """
    hits = [
        (path, line, text)
        for path, line, text in _scan(r"shift\(\s*-", root=SRC_DIR / "features")
        if path not in FEATURE_NEGATIVE_SHIFT_ALLOWLIST
    ]
    assert not hits, (
        "negative shift inside src/features/ (a feature is reaching into the "
        "future):\n"
        + "\n".join(f"  {path}:{line}: {text}" for path, line, text in hits)
    )


def test_no_unreviewed_negative_shifts_anywhere_in_src():
    """The same scan widened to the whole tree, with a documented allowlist.

    Stricter than the feature-only scan above: a backwards shift in a model, a
    metric or a backtest is just as capable of leaking. Every allowlisted path
    carries a written argument for why it is causal, next to the allowlist
    definition. A new negative shift anywhere fails here until someone makes
    that argument.
    """
    hits = [
        (path, line, text)
        for path, line, text in _scan(r"shift\(\s*-")
        if path not in NEGATIVE_SHIFT_ALLOWLIST
    ]
    assert not hits, (
        "negative shift outside the reviewed allowlist "
        f"{sorted(NEGATIVE_SHIFT_ALLOWLIST)}:\n"
        + "\n".join(f"  {path}:{line}: {text}" for path, line, text in hits)
        + "\nEither the shift is a leak, or it is causal and the reason belongs "
        "in NEGATIVE_SHIFT_ALLOWLIST in this file."
    )


def test_negative_shift_allowlist_has_no_dead_entries():
    """Every allowlisted path must exist and must actually contain a shift.

    Stops the allowlist rotting into a list of paths that no longer exist,
    which would silently start permitting a file that does.
    """
    hits = _scan(r"shift\(\s*-")
    seen = {path for path, _, _ in hits}
    dead = sorted(NEGATIVE_SHIFT_ALLOWLIST - seen)
    assert not dead, (
        f"these paths are allowlisted for negative shifts but no longer "
        f"contain one: {dead}. Remove them so the allowlist keeps meaning "
        "what it says."
    )


def test_the_negative_shift_scan_actually_finds_something():
    """Positive control for the grep above.

    If the regex or the file walk silently matched nothing at all, the test
    above would pass on an empty result set and prove nothing. The targets
    module is known to contain negative shifts, so it must show up.
    """
    hits = _scan(r"shift\(\s*-")
    allowed = [h for h in hits if h[0] in NEGATIVE_SHIFT_ALLOWLIST]
    assert allowed, (
        "the negative-shift scan found no matches at all, not even the ones "
        "src/data/targets.py is known to contain. The scan is broken."
    )


# ---------------------------------------------------------------------------
# Splitter: ordering, purge, embargo
# ---------------------------------------------------------------------------


def test_splitter_produces_folds_at_all(synthetic_panel, base_config):
    """Guard rail: every splitter test below is vacuous if there are no folds."""
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    assert len(folds) >= 3, f"expected several folds on the synthetic panel, got {len(folds)}"


def test_train_always_ends_before_test_begins(synthetic_panel, base_config):
    """max(train date) < min(test date) for every fold.

    Catches a splitter that shuffles rows or that builds the test window out of
    dates interleaved with the training window, which is the most blatant form
    of look-ahead.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    dates = _dates_of(matrix.X.index)
    _, folds = _folds(matrix.X.index)

    for fold in folds:
        train_dates = dates[fold.train_mask.to_numpy()]
        test_dates = dates[fold.test_mask.to_numpy()]
        assert train_dates.max() < test_dates.min(), (
            f"fold {fold.fold_id}: last training date {train_dates.max()} is not "
            f"before the first test date {test_dates.min()}"
        )


def test_no_row_appears_in_more_than_one_split(synthetic_panel, base_config):
    """train, val and test must be disjoint within a fold.

    Catches an off-by-one in the mask arithmetic that would let the same
    (asset, date) row be trained on and scored on.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)

    for fold in folds:
        train = fold.train_mask.to_numpy()
        val = fold.val_mask.to_numpy()
        test = fold.test_mask.to_numpy()
        overlaps = {
            "train/val": int((train & val).sum()),
            "train/test": int((train & test).sum()),
            "val/test": int((val & test).sum()),
        }
        assert not any(overlaps.values()), (
            f"fold {fold.fold_id} has rows in more than one split: {overlaps}"
        )


@pytest.mark.parametrize("embargo_days", [0, 5, 20])
def test_purge_gap_is_at_least_horizon_plus_embargo(
    synthetic_panel, base_config, embargo_days
):
    """The gap between the last fitting row and the first test row.

    A label at date d is a fact about d + horizon, so a training row inside
    ``horizon`` days of the test window has a label drawn from the test period.
    The embargo adds a further buffer on top. embargo_days=0 is included to
    confirm the purge alone still removes the horizon; if the purge were
    implemented as part of the embargo, that case would collapse to a zero gap.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    splitter, folds = _folds(matrix.X.index, embargo_days=embargo_days)
    required = splitter.horizon + embargo_days

    for fold in folds:
        fit_end = _fit_end(matrix.X.index, fold)
        gap = (fold.test_start - fit_end).days
        assert gap >= required, (
            f"fold {fold.fold_id} with embargo_days={embargo_days}: only {gap} "
            f"calendar days between the last fitting row ({fit_end.date()}) and "
            f"the first test row ({fold.test_start.date()}), need at least "
            f"{required} (horizon {splitter.horizon} + embargo {embargo_days})"
        )


@pytest.mark.parametrize("embargo_days", [0, 5, 20])
def test_fold_reports_the_embargo_it_actually_applied(
    synthetic_panel, base_config, embargo_days
):
    """n_embargoed must be zero when the embargo is off and positive otherwise.

    Reads the mechanism directly rather than inferring it from row counts, so
    a splitter that quietly ignored the embargo setting cannot hide behind the
    purge already having removed some rows.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index, embargo_days=embargo_days)

    for fold in folds:
        if embargo_days == 0:
            assert fold.n_embargoed == 0, (
                f"fold {fold.fold_id} embargoed {fold.n_embargoed} rows with "
                "embargo_days=0"
            )
        else:
            assert fold.n_embargoed > 0, (
                f"fold {fold.fold_id} embargoed nothing with "
                f"embargo_days={embargo_days}; the setting is being ignored"
            )


def test_increasing_the_embargo_strictly_shrinks_the_training_data(
    synthetic_panel, base_config
):
    """More embargo must mean fewer fitting rows for the same fold.

    Measured on train + val, which is exactly the set of rows surviving the
    embargo boundary. Measuring train alone would also move because the inner
    validation cut shifts, making the comparison empirical rather than
    structural. Catches an embargo_days argument that is accepted and dropped.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)

    counts: dict[int, list[int]] = {}
    for embargo_days in (0, 5, 20):
        _, folds = _folds(matrix.X.index, embargo_days=embargo_days)
        counts[embargo_days] = [fold.n_train + fold.n_val for fold in folds]

    assert len({len(v) for v in counts.values()}) == 1, (
        f"the embargo changed the number of folds: "
        f"{ {k: len(v) for k, v in counts.items()} }; the comparison below "
        "would not be like for like"
    )

    for fold_id, (none, medium, large) in enumerate(
        zip(counts[0], counts[5], counts[20])
    ):
        assert none > medium > large, (
            f"fold {fold_id}: fitting rows for embargo 0/5/20 are "
            f"{none}/{medium}/{large}; a larger embargo did not remove more rows"
        )


def test_inner_validation_boundary_is_purged_too(synthetic_panel, base_config):
    """max(train date) < min(val date), with the same gap as the outer boundary.

    Early stopping reads the validation loss. If the inner boundary is not
    purged, the epoch chosen is tuned on rows whose labels overlap the training
    data, and the model is silently selected on information it should not have.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    dates = _dates_of(matrix.X.index)
    embargo_days = 5
    splitter, folds = _folds(matrix.X.index, embargo_days=embargo_days)
    required = splitter.horizon + embargo_days

    for fold in folds:
        train_dates = dates[fold.train_mask.to_numpy()]
        val_dates = dates[fold.val_mask.to_numpy()]
        assert len(val_dates) > 0, f"fold {fold.fold_id} has an empty validation slice"
        assert train_dates.max() < val_dates.min(), (
            f"fold {fold.fold_id}: inner train ends {train_dates.max()} which is "
            f"not before validation starts {val_dates.min()}"
        )
        gap = (val_dates.min() - train_dates.max()).days
        assert gap >= required, (
            f"fold {fold.fold_id}: only {gap} days between inner train and "
            f"validation, need at least {required}"
        )


# ---------------------------------------------------------------------------
# Preprocessing: fit on train, and only on train
# ---------------------------------------------------------------------------


def _fold_frames(matrix, fold):
    train = matrix.X[fold.train_mask.to_numpy()]
    val = matrix.X[fold.val_mask.to_numpy()]
    test = matrix.X[fold.test_mask.to_numpy()]
    return train, val, test


def test_scaler_is_refit_for_every_fold(synthetic_panel, base_config):
    """Two different folds must learn different means.

    Identical means across folds is the fingerprint of a scaler that was fit
    once on the whole panel and reused, which pushes late-sample volatility
    backwards into early training data.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    assert len(folds) >= 3

    first_train, _, _ = _fold_frames(matrix, folds[0])
    later_train, _, _ = _fold_frames(matrix, folds[2])

    first = FoldPreprocessor.from_config(base_config).fit(first_train)
    later = FoldPreprocessor.from_config(base_config).fit(later_train)

    assert len(first_train) != len(later_train), (
        "the two folds have the same training size, so this test would not "
        "distinguish a per-fold fit from a global one"
    )
    same = np.allclose(
        first.means_.to_numpy(), later.means_.to_numpy(), equal_nan=True
    )
    assert not same, (
        "fold 0 and fold 2 learned identical means. The scaler is being fit "
        "once globally instead of inside each fold."
    )


def test_scaler_fit_row_count_equals_the_training_rows(synthetic_panel, base_config):
    """n_fit_rows_ must be the training count, never the whole panel.

    A blunt but decisive check: if the preprocessor was handed the full feature
    matrix by mistake, this number is the panel length.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)

    for fold in folds:
        train, _, _ = _fold_frames(matrix, fold)
        preprocessor = FoldPreprocessor.from_config(base_config).fit(train)
        assert preprocessor.n_fit_rows_ == fold.n_train, (
            f"fold {fold.fold_id}: preprocessor fit on "
            f"{preprocessor.n_fit_rows_} rows but the fold has "
            f"{fold.n_train} training rows"
        )
        assert preprocessor.n_fit_rows_ < len(matrix.X), (
            f"fold {fold.fold_id}: the preprocessor saw every row of the panel "
            f"({len(matrix.X)}), not just the training fold"
        )


def test_scaler_statistics_change_when_test_rows_are_added(synthetic_panel, base_config):
    """Fitting on train vs train+test must give different statistics.

    If they came out the same, the "train only" fit was not actually excluding
    anything and the whole per-fold discipline would be an illusion.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    fold = folds[0]
    train, _, test = _fold_frames(matrix, fold)

    honest = FoldPreprocessor.from_config(base_config).fit(train)
    leaky = FoldPreprocessor.from_config(base_config).fit(pd.concat([train, test]))

    assert leaky.n_fit_rows_ == len(train) + len(test)
    for name in ("means_", "stds_", "lower_bounds_", "upper_bounds_"):
        honest_values = getattr(honest, name).to_numpy()
        leaky_values = getattr(leaky, name).to_numpy()
        assert not np.allclose(honest_values, leaky_values, equal_nan=True), (
            f"{name} is unchanged when the test rows are added to the fit. "
            "Either the fit is ignoring its input or the test rows are already "
            "inside the training frame."
        )


def test_transformed_test_rows_do_not_look_like_they_were_fit_on(
    synthetic_panel, base_config
):
    """Test rows scaled by a train-fitted scaler must not be exactly standardised.

    Mean 0 and standard deviation 1 on the test rows is the signature of a
    scaler that saw them during fit. The positive control below shows what that
    signature looks like, so the assertion is a comparison against a known
    fingerprint rather than an arbitrary tolerance.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    fold = folds[0]
    train, _, test = _fold_frames(matrix, fold)

    honest = FoldPreprocessor.from_config(base_config).fit(train)
    scaled = honest.transform(test)

    # Positive control: a scaler fit on the test rows themselves produces a
    # mean of essentially exactly zero on those rows.
    leaky = FoldPreprocessor.from_config(base_config)
    leaky_scaled = leaky.fit_transform(test)
    leaky_mean = float(leaky_scaled.mean().abs().max())
    assert leaky_mean < 1e-8, (
        f"the test is broken: a scaler fit directly on the test rows still "
        f"leaves a mean of {leaky_mean:.3e}, so the fingerprint below is not "
        "what a leak actually looks like"
    )

    means_zero = np.allclose(scaled.mean().to_numpy(), 0.0, atol=1e-8)
    stds_one = np.allclose(scaled.std(ddof=0).to_numpy(), 1.0, atol=1e-8)
    assert not (means_zero and stds_one), (
        "the transformed test rows have mean 0 and standard deviation 1 to "
        "machine precision. The scaler was fit on the test data."
    )

    observed = float(scaled.mean().abs().max())
    assert observed > 1e-6, (
        f"the largest absolute column mean on the scaled test rows is "
        f"{observed:.3e}, indistinguishable from a scaler that was fit on them"
    )


def test_winsorisation_bounds_are_the_train_quantiles_and_bind_on_test(
    synthetic_panel, base_config
):
    """Clip bounds come from the training quantiles, and they actually clip.

    Two failure modes. If the bounds were computed on train+test, extreme test
    moves would set the bounds and no test value would ever fall outside them.
    If the clip were not applied at transform time, an outlier in the test set
    would pass through at full size.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    fold = folds[0]
    train, _, test = _fold_frames(matrix, fold)

    preprocessor = FoldPreprocessor.from_config(base_config).fit(train)
    assert preprocessor.winsorize, "winsorisation is disabled in config/base.yaml"

    pd.testing.assert_series_equal(
        preprocessor.lower_bounds_,
        train.quantile(preprocessor.lower_q),
        check_names=False,
        obj="lower_bounds_",
    )
    pd.testing.assert_series_equal(
        preprocessor.upper_bounds_,
        train.quantile(preprocessor.upper_q),
        check_names=False,
        obj="upper_bounds_",
    )

    below = int((test < preprocessor.lower_bounds_).to_numpy().sum())
    above = int((test > preprocessor.upper_bounds_).to_numpy().sum())
    assert below > 0 and above > 0, (
        f"only {below} test values below and {above} above the train clip "
        "bounds. If nothing on the test side is ever clipped, the bounds were "
        "almost certainly computed with the test rows included."
    )

    # And the clip must be applied, not merely recorded.
    clipped = test.clip(preprocessor.lower_bounds_, preprocessor.upper_bounds_, axis=1)
    scaled = preprocessor.transform(test)
    expected = (clipped.fillna(preprocessor.medians_) - preprocessor.means_) / preprocessor.stds_
    expected = expected.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    pd.testing.assert_frame_equal(scaled, expected, obj="transform output")


# ---------------------------------------------------------------------------
# Feature causality, end to end
# ---------------------------------------------------------------------------


def _truncation_diff(panel: pd.DataFrame, config: dict, cut_position: int):
    """Build features on the full panel and on a truncated copy, then compare.

    Returns the per-column maximum absolute difference over the rows both
    builds should agree on.
    """
    unique_dates = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(panel.index.get_level_values(schema.DATE)).unique())
    )
    truncate_at = unique_dates[cut_position]

    full = build_features(panel, config).X
    truncated = build_features(
        panel[panel.index.get_level_values(schema.DATE) <= truncate_at], config
    ).X

    lookback = int(config["features"]["max_lookback"])
    safe_until = truncate_at - pd.Timedelta(days=lookback)

    left = full[full.index.get_level_values(schema.DATE) <= safe_until]
    right = truncated[truncated.index.get_level_values(schema.DATE) <= safe_until]
    return left, right, truncate_at, safe_until


def test_features_do_not_change_when_later_data_is_removed(
    synthetic_panel, base_config
):
    """Rebuilding on a truncated panel must reproduce the earlier rows exactly.

    This is the end-to-end causality test. Every feature here is backward
    looking, so deleting everything after T cannot change any value at or
    before T. If a feature reached forward, even through something indirect
    like a full-sample quantile or a centered window, its earlier values would
    move when the future is taken away. The comparison window stops at
    T - max_lookback so that no partially filled long window is compared.
    """
    left, right, truncate_at, safe_until = _truncation_diff(
        synthetic_panel, base_config, cut_position=600
    )

    assert left.index.equals(right.index), (
        f"the two builds do not even cover the same rows up to "
        f"{safe_until.date()}: {len(left)} vs {len(right)}"
    )
    assert len(left) > 0, "the comparison window is empty; this test proves nothing"

    difference = (left - right).abs().max()
    worst = difference.sort_values(ascending=False)
    # Exact equality, not a tolerance. These are the same arithmetic operations
    # on the same inputs, so any nonzero difference means a feature saw data
    # from after the truncation point.
    assert float(worst.iloc[0]) == 0.0, (
        f"features changed when data after {truncate_at.date()} was removed, so "
        "they are reaching forward in time. Worst offenders:\n"
        + worst.head(10).to_string()
    )


# ---------------------------------------------------------------------------
# Baselines are scored on the same rows as the models
# ---------------------------------------------------------------------------


def _fold_data(matrix, fold, target: str) -> FoldData:
    train_mask = fold.train_mask.to_numpy()
    val_mask = fold.val_mask.to_numpy()
    test_mask = fold.test_mask.to_numpy()
    return FoldData(
        fold_id=fold.fold_id,
        X_train=matrix.X[train_mask],
        y_train=matrix.targets[target][train_mask],
        X_val=matrix.X[val_mask],
        y_val=matrix.targets[target][val_mask],
        X_test=matrix.X[test_mask],
        y_test=matrix.targets[target][test_mask],
        meta_train=matrix.meta[train_mask],
        meta_test=matrix.meta[test_mask],
        train_start=fold.train_start,
        train_end=fold.train_end,
        test_start=fold.test_start,
        test_end=fold.test_end,
        target=target,
    )


def test_baselines_predict_on_exactly_the_fold_test_index(
    synthetic_panel, base_config
):
    """Every baseline returns one prediction per test row, on the same index.

    A baseline that silently drops NaN rows would be scored on an easier subset
    than the models it is compared against, and the comparison that the whole
    report rests on would be between two different populations.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    target = base_config["targets"]["primary"]
    _, folds = _folds(matrix.X.index)
    fold = folds[0]
    fold_data = _fold_data(matrix, fold, target)

    # Built independently of FoldData so a bug inside it cannot make this pass.
    expected_index = matrix.X.index[fold.test_mask.to_numpy()]

    baselines = build_baselines(target, task=CLASSIFICATION)
    assert baselines, f"no baselines are registered for target {target!r}"

    for model in baselines:
        prediction = model.run_fold(fold_data)
        assert prediction.index.equals(expected_index), (
            f"baseline {model.name!r} predicted on a different index than the "
            f"fold's test rows ({len(prediction.index)} vs "
            f"{len(expected_index)} rows)"
        )
        assert len(prediction.point) == len(expected_index), (
            f"baseline {model.name!r} returned {len(prediction.point)} "
            f"predictions for {len(expected_index)} test rows"
        )
        assert np.isfinite(prediction.point).all(), (
            f"baseline {model.name!r} produced non-finite predictions; those "
            "rows would silently drop out of the metric"
        )


def test_baselines_never_see_a_target_column_through_meta(
    synthetic_panel, base_config
):
    """meta reaches a model stripped of every target and forward-looking column.

    ``meta`` legitimately carries close, high, low and a lagged return, and a
    model is allowed those. It must never carry ret_1d, dir_1d, vol_1d,
    fwd_ret_simple or realised_var_1d, any one of which is the answer.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    target = base_config["targets"]["primary"]
    _, folds = _folds(matrix.X.index)
    fold_data = _fold_data(matrix, folds[0], target)

    # Positive control: the raw meta really does contain the answer, so the
    # stripping step below has something to do.
    assert target in fold_data.meta_test.columns, (
        "meta does not carry the target at all, so this test proves nothing"
    )

    model = build_baselines(target, task=CLASSIFICATION)[0]
    safe = model._safe_meta(fold_data.meta_test)

    forbidden = [
        column
        for column in safe.columns
        if column in ("ret_1d", "dir_1d", "vol_1d")
        or column.startswith("fwd_")
        or column.startswith("realised_var_1d")
    ]
    assert not forbidden, f"a target column survived _safe_meta: {forbidden}"

    # The columns a model is entitled to must still be there.
    for column in (schema.CLOSE, "ret_lag1", "realised_var_now"):
        assert column in safe.columns, f"_safe_meta removed the legitimate column {column!r}"


def test_feature_matrix_never_contains_a_target_column(synthetic_panel, base_config):
    """X and targets must be disjoint.

    Catches a target accidentally concatenated into the feature block, which
    would give a model the answer as an input column.
    """
    matrix = _synthetic_matrix(synthetic_panel, base_config)
    overlap = set(matrix.X.columns) & set(matrix.targets.columns)
    assert not overlap, f"target columns present in the feature matrix: {sorted(overlap)}"

    forward = [c for c in matrix.X.columns if c.startswith("fwd_") or c.startswith("realised_var_1d")]
    assert not forward, f"forward-looking columns present in the feature matrix: {forward}"


# ---------------------------------------------------------------------------
# The GARCH recursion is causal
# ---------------------------------------------------------------------------


def _fitted_garch(synthetic_panel, base_config):
    """A GARCH fitted on the last synthetic fold, together with that fold.

    The last fold is used because it has the longest training window, and the
    model refuses to fit an asset with fewer than 250 usable training returns.
    """
    from src.models.garch import GarchModel

    matrix = _synthetic_matrix(synthetic_panel, base_config)
    _, folds = _folds(matrix.X.index)
    fold_data = _fold_data(matrix, folds[-1], "vol_1d")

    model = GarchModel(params={"variant": "garch11"}, task=REGRESSION)
    model.fit(fold_data)
    assert model.fitted_params_, (
        "GARCH fitted no asset on this fold, so every assertion below would be "
        "comparing one constant fallback against another and would pass no "
        "matter what the recursion did"
    )
    return model, fold_data


def _garch_predictions(model, meta: pd.DataFrame) -> np.ndarray:
    safe = model._safe_meta(meta)
    return np.asarray(model.predict(safe, safe), dtype=float)


def test_garch_recursion_reproduces_arch_on_the_training_window(
    synthetic_panel, base_config
):
    """The hand-driven recursion is arch's recursion, not an approximation of it.

    src/models/garch.py runs the conditional variance recursion itself so that
    the backcast and the variance bounds can be taken from the training prefix
    alone. That freedom is only worth having if the recursion it runs is the one
    arch would have run: a wrong parameter order or a mishandled mean produces
    numbers that are wrong and perfectly plausible, which is exactly the failure
    this model has already been bitten by once.

    The two paths are not expected to agree to machine precision at the very
    start of the sample, because arch seeds its backcast from the residuals of a
    starting-value mean while this one uses the fitted mean. That difference
    decays geometrically in beta, so the comparison starts after a burn-in.
    """
    model, _ = _fitted_garch(synthetic_panel, base_config)
    burn_in = 100

    for asset, params in model.fitted_params_.items():
        train = model.train_series_[asset]
        assert len(train) > burn_in + 100, "training window too short to burn in"

        ours = model._conditional_variance(params, train, len(train))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            theirs = (
                np.asarray(
                    model._make_model(train).fix(params).conditional_volatility,
                    dtype=float,
                )
                ** 2
            )

        relative = np.abs(ours - theirs) / theirs
        assert relative[burn_in:].max() < 1e-4, (
            f"the recursion for {asset} diverges from arch's own by "
            f"{relative[burn_in:].max():.2e} after burn-in, so it is not "
            "computing the model that was fitted"
        )


@pytest.mark.parametrize("cut", [0, 30])
def test_garch_forecasts_do_not_change_when_later_data_changes(
    synthetic_panel, base_config, cut
):
    """The forecast for date t is a function of returns up to t and nothing else.

    This is the assertion that makes the shift(-1) in src/models/garch.py
    defensible. The recursion is run over the training and test returns
    together, so nothing short of a direct test rules out the test tail leaking
    backwards into an earlier forecast.

    It does leak if the recursion is left to arch's fix(): fix() computes the
    backcast, the demeaning constant behind it and the variance bounds over the
    whole series it is handed, and multiplying the tail of the test window by
    twenty moves the variance at the start of that recursion by around two
    percent. The influence has decayed to nothing by the test window on this
    panel, which is how the module got away with it, but "too small to matter"
    is a different claim from "does not happen".

    ``cut`` is run at the first test date as well as in the middle, because a
    version that miscounts the training prefix by a few rows still passes the
    middle-of-the-window case.
    """
    model, fold_data = _fitted_garch(synthetic_panel, base_config)
    meta_test = fold_data.meta_test
    dates = _dates_of(meta_test.index).unique().sort_values()
    assert len(dates) > cut + 5, "test window too short for this cut"
    boundary = dates[cut]

    baseline = _garch_predictions(model, meta_test)
    assert np.ptp(baseline) > 0, (
        "every GARCH forecast on this fold is identical, so the model is "
        "returning its constant fallback and this test proves nothing"
    )

    perturbed = meta_test.copy()
    later = _dates_of(perturbed.index) > boundary
    assert later.any(), "nothing after the cut to perturb"
    perturbed.loc[later, "ret_lag1"] = perturbed.loc[later, "ret_lag1"] * 20.0

    after = _garch_predictions(model, perturbed)

    unchanged = _dates_of(meta_test.index) <= boundary
    assert np.array_equal(baseline[unchanged], after[unchanged]), (
        "a GARCH forecast dated on or before "
        f"{boundary.date()} moved when returns after that date were changed, "
        f"by up to {np.abs(baseline[unchanged] - after[unchanged]).max():.3e} "
        "in log variance: the recursion is reading the future"
    )
    assert not np.array_equal(baseline[~unchanged], after[~unchanged]), (
        "changing the returns after the cut changed no forecast at all, so the "
        "perturbation never reached the model and the assertion above is "
        "vacuous"
    )


# ---------------------------------------------------------------------------
# Real panel checks. Skipped when data/interim/panel_1d.parquet is absent.
# ---------------------------------------------------------------------------


@requires_real_panel
def test_real_panel_splitter_ordering_and_purge(real_panel, base_config):
    """The ordering and purge guarantees on the real folds, from base.yaml.

    Same assertions as the synthetic versions, but driven by the settings that
    actually produce the report: expanding scheme, quarterly test windows,
    five day embargo, first test window in 2020.
    """
    matrix = build_features(real_panel, base_config)
    splitter = splitter_from_config(base_config)
    folds = list(splitter.split(matrix.X.index))
    assert len(folds) > 10, f"expected many folds on nine years of data, got {len(folds)}"

    dates = _dates_of(matrix.X.index)
    required = splitter.horizon + splitter.embargo_days

    for fold in folds:
        train_dates = dates[fold.train_mask.to_numpy()]
        val_dates = dates[fold.val_mask.to_numpy()]
        test_dates = dates[fold.test_mask.to_numpy()]

        assert train_dates.max() < test_dates.min(), (
            f"real fold {fold.fold_id}: train ends after test begins"
        )
        assert train_dates.max() < val_dates.min(), (
            f"real fold {fold.fold_id}: inner train ends after validation begins"
        )

        gap = (fold.test_start - _fit_end(matrix.X.index, fold)).days
        assert gap >= required, (
            f"real fold {fold.fold_id}: purge gap is {gap} days, need {required}"
        )

        overlaps = int(
            (fold.train_mask & fold.test_mask).sum()
            + (fold.val_mask & fold.test_mask).sum()
            + (fold.train_mask & fold.val_mask).sum()
        )
        assert overlaps == 0, f"real fold {fold.fold_id}: {overlaps} overlapping rows"


@requires_real_panel
def test_real_panel_scaler_is_refit_per_fold(real_panel, base_config):
    """Different real folds must learn different scaling statistics."""
    matrix = build_features(real_panel, base_config)
    splitter = splitter_from_config(base_config)
    folds = list(splitter.split(matrix.X.index))

    first = FoldPreprocessor.from_config(base_config).fit(
        matrix.X[folds[0].train_mask.to_numpy()]
    )
    last = FoldPreprocessor.from_config(base_config).fit(
        matrix.X[folds[-1].train_mask.to_numpy()]
    )

    assert first.n_fit_rows_ == folds[0].n_train
    assert last.n_fit_rows_ == folds[-1].n_train
    assert not np.allclose(
        first.means_.to_numpy(), last.means_.to_numpy(), equal_nan=True
    ), "the first and last real folds learned identical means"


@requires_real_panel
def test_real_panel_features_do_not_change_when_later_data_is_removed(
    real_panel, base_config
):
    """The end-to-end causality test on the real panel.

    Truncated part way through 2022, which is inside the SOL history as well as
    the three older assets, so the cross-asset family is exercised with a full
    complement of assets on both sides of the comparison.
    """
    unique_dates = pd.DatetimeIndex(
        sorted(pd.DatetimeIndex(real_panel.index.get_level_values(schema.DATE)).unique())
    )
    cut_position = len(unique_dates) // 2
    left, right, truncate_at, safe_until = _truncation_diff(
        real_panel, base_config, cut_position=cut_position
    )

    assert left.index.equals(right.index), (
        f"real panel: the two builds cover different rows up to {safe_until.date()}"
    )
    assert len(left) > 1000, "real panel: comparison window is suspiciously small"

    worst = (left - right).abs().max().sort_values(ascending=False)
    assert float(worst.iloc[0]) == 0.0, (
        f"real panel: features changed when data after {truncate_at.date()} was "
        "removed. Worst offenders:\n" + worst.head(10).to_string()
    )


@requires_real_panel
def test_real_panel_targets_are_never_inside_the_feature_matrix(real_panel, base_config):
    """No target or forward-looking column may appear in X on the real panel."""
    matrix = build_features(real_panel, base_config)
    targets = build_targets(real_panel)
    overlap = set(matrix.X.columns) & set(targets.columns)
    assert not overlap, f"real panel: target columns inside X: {sorted(overlap)}"
