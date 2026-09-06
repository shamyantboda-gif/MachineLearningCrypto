"""Per fold metrics and the aggregation the report layer consumes.

Two conventions run through this module.

First, degenerate folds return nan instead of raising. A walk-forward split
produces short test windows, and some of them contain a single class or too few
rows to score. Those folds carry no information about skill, but they must not
abort a run that has twenty informative folds behind them. The nan then
propagates visibly into the results table instead of hiding as a silent zero.

Second, nothing here uses test set statistics as a reference point. That rules
out the test set mean as the R-squared benchmark, and it is why
:func:`r2_oos` defaults to a zero forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import balanced_accuracy_score as _sk_balanced_accuracy
from sklearn.metrics import brier_score_loss as _sk_brier
from sklearn.metrics import matthews_corrcoef as _sk_mcc
from sklearn.metrics import roc_auc_score as _sk_roc_auc

from src.models.base import CLASSIFICATION, REGRESSION, Prediction

#: Extra dispatch value for :func:`evaluate_predictions`. ``vol_1d`` is a
#: REGRESSION target in ``TARGET_TASKS``, so the volatility diagnostics are
#: opt in: the caller has to confirm it has converted to the variance scale.
VOLATILITY = "volatility"

NAN = float("nan")


def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten to float arrays and drop rows where either side is missing.

    Dropping jointly keeps the two series aligned. Scoring a model on the rows
    where it happened to produce a number, against a longer truth series, is a
    subtle way to flatter it.
    """
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} vs {y.shape}")
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. nan on an empty sample."""
    y, f = _clean_pair(y_true, y_pred)
    if y.size == 0:
        return NAN
    return float(np.sqrt(np.mean((y - f) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. nan on an empty sample."""
    y, f = _clean_pair(y_true, y_pred)
    if y.size == 0:
        return NAN
    return float(np.mean(np.abs(y - f)))


def r2_oos(
    y_true: np.ndarray, y_pred: np.ndarray, benchmark: np.ndarray | None = None
) -> float:
    """Out of sample R-squared against an explicit benchmark forecast.

    Returns ``1 - SSE_model / SSE_benchmark``. The benchmark defaults to a zero
    forecast, not the test set mean. The test set mean is not knowable at
    forecast time, so scoring against it credits the model with information it
    could not have had and inflates the number. Zero is the honest reference
    for a daily return series: it is what a forecaster with no view would say,
    and it is available in advance.

    Negative values are the normal outcome on daily crypto returns and mean the
    model does worse than that no view forecast. Pass ``benchmark`` explicitly
    to score against something else, for example a fitted random walk drift.
    """
    y_raw = np.asarray(y_true, dtype=float).ravel()
    f_raw = np.asarray(y_pred, dtype=float).ravel()
    if y_raw.shape != f_raw.shape:
        raise ValueError(f"length mismatch: {y_raw.shape} vs {f_raw.shape}")
    if benchmark is None:
        bench_raw = np.zeros_like(y_raw)
    else:
        bench_raw = np.asarray(benchmark, dtype=float).ravel()
        if bench_raw.shape != y_raw.shape:
            raise ValueError("benchmark must be the same length as y_true")

    # One mask across all three series so model and benchmark are scored on
    # exactly the same rows.
    mask = np.isfinite(y_raw) & np.isfinite(f_raw) & np.isfinite(bench_raw)
    y, f, bench = y_raw[mask], f_raw[mask], bench_raw[mask]
    if y.size == 0:
        return NAN
    sse_bench = float(np.sum((y - bench) ** 2))
    if sse_bench <= 0.0:
        # An all zero truth series against a zero benchmark leaves nothing to
        # explain, so the ratio is undefined rather than perfect.
        return NAN
    sse_model = float(np.sum((y - f) ** 2))
    return 1.0 - sse_model / sse_bench


# ---------------------------------------------------------------------------
# Classification on P(up)
# ---------------------------------------------------------------------------


def _clean_classification(
    y_true: np.ndarray, score: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return integer labels in {0, 1} and the aligned score array."""
    y, s = _clean_pair(y_true, score)
    # The panel encodes dir_1d as 1 for up and 0 otherwise. The ``> 0`` cast
    # also survives a {-1, 1} encoding, which would otherwise be scored as if
    # every down day were an up day.
    return (y > 0).astype(int), s


def _scoreable(labels: np.ndarray) -> bool:
    """True when the fold can support a classification metric.

    A single class fold is treated as unscoreable across the board, including
    for accuracy and Brier score, which are arithmetically defined there. On an
    all up fold a model that always says up scores 100 percent while knowing
    nothing, so reporting that number next to a base rate of 1.0 invites the
    exact misreading the project is trying to avoid.
    """
    return labels.size > 0 and np.unique(labels).size >= 2


def directional_accuracy(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Fraction of days whose direction was called correctly.

    Read this only against :func:`base_rate` from the same fold. nan when the
    fold is empty or single class.
    """
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return NAN
    return float(np.mean((s >= threshold).astype(int) == labels))


def base_rate(y_true: np.ndarray, proba: np.ndarray | None = None) -> float:
    """Fraction of up days in the fold, the accuracy of always predicting up.

    nan when the fold is empty or single class, matching the other
    classification metrics so a degenerate fold reads as degenerate across the
    whole row rather than showing a lone 1.0 that looks like a score.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    y = y[np.isfinite(y)]
    labels = (y > 0).astype(int)
    if not _scoreable(labels):
        return NAN
    return float(np.mean(labels))


def balanced_accuracy(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Mean of sensitivity and specificity. nan on an empty or single class fold.

    The guard is an up front check rather than a try/except because sklearn
    returns 0.5 here with a warning instead of raising, and a silently wrong
    0.5 is worse than a nan.
    """
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return NAN
    return float(_sk_balanced_accuracy(labels, (s >= threshold).astype(int)))


def matthews_corrcoef(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> float:
    """Matthews correlation coefficient. nan on an empty or single class fold.

    As with balanced accuracy the guard is up front: sklearn returns 0.0 on a
    single class input, which is indistinguishable from a genuine no skill
    result on a healthy fold.
    """
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return NAN
    return float(_sk_mcc(labels, (s >= threshold).astype(int)))


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Area under the ROC curve on P(up). nan on an empty or single class fold."""
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return NAN
    return float(_sk_roc_auc(labels, s))


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Mean squared error of the probability forecast. Lower is better.

    nan on an empty or single class fold, for the same reason as
    :func:`directional_accuracy`.
    """
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return NAN
    return float(_sk_brier(labels, s))


CALIBRATION_COLUMNS = ["bin_mid", "mean_pred", "frac_positive", "count"]


def calibration_curve_data(
    y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Reliability table for P(up), one row per non empty probability bin.

    Bins are equal width over [0, 1]. Empty bins are dropped rather than
    emitted with a nan mean, so the row ``count`` column is always positive and
    a caller can weight by it without filtering first.

    Being the one function here that returns a table, the degenerate answer is
    an empty frame carrying the declared columns rather than a scalar nan. An
    empty or single class fold therefore yields ``len(result) == 0``.
    """
    labels, s = _clean_classification(y_true, proba)
    if not _scoreable(labels):
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in CALIBRATION_COLUMNS})

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Clip so probabilities sitting exactly on 0 or 1 land in the end bins
    # instead of falling outside the range.
    idx = np.clip(np.digitize(s, edges[1:-1], right=False), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = idx == b
        count = int(in_bin.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_mid": float((edges[b] + edges[b + 1]) / 2.0),
                "mean_pred": float(np.mean(s[in_bin])),
                "frac_positive": float(np.mean(labels[in_bin])),
                "count": count,
            }
        )
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in CALIBRATION_COLUMNS})
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def qlike(y_true_var: np.ndarray, pred_var: np.ndarray) -> float:
    """Mean QLIKE loss, the standard volatility forecast loss. Lower is better.

    ``L = y / f - log(y / f) - 1``, which is zero for a perfect forecast and
    positive otherwise. Both arguments must be strictly positive VARIANCES.
    Passing log variances or standard deviations is a silent scale error that
    still returns a plausible number, so non-positive input raises rather than
    returning nan. Convert first: a ``vol_1d`` target stored as log volatility
    becomes a variance as ``np.exp(2 * y)``.
    """
    y, f = _clean_pair(y_true_var, pred_var)
    if y.size == 0:
        return NAN
    if np.any(y <= 0.0):
        raise ValueError(
            "qlike requires strictly positive realised variance; pass variances, "
            "not log variances or standard deviations"
        )
    if np.any(f <= 0.0):
        raise ValueError(
            "qlike requires strictly positive forecast variance; pass variances, "
            "not log variances or standard deviations"
        )
    ratio = y / f
    return float(np.mean(ratio - np.log(ratio) - 1.0))


@dataclass(frozen=True)
class MincerZarnowitz:
    """OLS of realised on predicted, plus the joint test of forecast optimality.

    ``r2`` is the in sample R-squared of that auxiliary regression, which is a
    goodness of fit for the realised/predicted relationship. It is not
    :func:`r2_oos` and the two should never be put in the same column.

    A small ``joint_p`` rejects ``alpha = 0 and beta = 1``, meaning the forecast
    is biased or mis-scaled even if it is correlated with the outcome.
    """

    alpha: float
    beta: float
    r2: float
    joint_p: float

    def __str__(self) -> str:
        return (
            f"MZ alpha={self.alpha:+.4g} beta={self.beta:+.4g} "
            f"r2={self.r2:.4f} joint_p={self.joint_p:.4f}"
        )


def mincer_zarnowitz(y_true: np.ndarray, y_pred: np.ndarray) -> MincerZarnowitz:
    """Regress realised on predicted and test alpha = 0, beta = 1 jointly.

    Fits ``y = alpha + beta * f + e`` by least squares and reports an F test of
    the two restrictions together, on ``F(2, n - 2)``. Implemented with numpy
    and scipy directly so the evaluation layer keeps no statsmodels dependency.

    Degenerate inputs return nan fields rather than raising: fewer than three
    observations, a constant forecast (the design matrix is singular), or a
    perfect fit leaving no residual variance to scale the test by.
    """
    y, f = _clean_pair(y_true, y_pred)
    n = int(y.size)
    if n < 3:
        return MincerZarnowitz(NAN, NAN, NAN, NAN)

    X = np.column_stack([np.ones(n), f])
    xtx = X.T @ X
    # A constant forecast makes the second column collinear with the intercept,
    # so there is no unique slope to report.
    if not np.isfinite(np.linalg.cond(xtx)) or np.linalg.cond(xtx) > 1e12:
        return MincerZarnowitz(NAN, NAN, NAN, NAN)

    xtx_inv = np.linalg.inv(xtx)
    beta_hat = xtx_inv @ (X.T @ y)
    alpha, beta = float(beta_hat[0]), float(beta_hat[1])

    resid = y - X @ beta_hat
    rss = float(resid @ resid)
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - rss / tss if tss > 0.0 else NAN

    s2 = rss / (n - 2)
    if s2 <= 0.0:
        return MincerZarnowitz(alpha, beta, r2, NAN)

    diff = beta_hat - np.array([0.0, 1.0])
    cov = xtx_inv * s2
    try:
        f_stat = float(diff @ np.linalg.solve(cov, diff) / 2.0)
    except np.linalg.LinAlgError:
        return MincerZarnowitz(alpha, beta, r2, NAN)
    if not np.isfinite(f_stat) or f_stat < 0.0:
        return MincerZarnowitz(alpha, beta, r2, NAN)
    joint_p = float(stats.f.sf(f_stat, 2, n - 2))
    return MincerZarnowitz(alpha, beta, r2, joint_p)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def evaluate_predictions(
    y_true: pd.Series,
    prediction: Prediction,
    task: str,
    benchmark: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one fold and return a flat row for ``reports/results/results.csv``.

    ``y_true`` is reindexed onto ``prediction.index`` before scoring. Both carry
    the (asset, date) MultiIndex, so aligning by label rather than by position
    is what stops a reordered or partially covered fold from producing
    confident nonsense.

    ``task`` is ``REGRESSION`` or ``CLASSIFICATION`` from ``src.models.base``,
    or the local ``VOLATILITY`` value. Volatility is opt in because ``vol_1d``
    is registered as a regression target and is stored on a log scale, while
    QLIKE and Mincer-Zarnowitz need variances. Ask for it only after converting.
    The volatility row omits ``r2_oos`` unless ``benchmark`` is supplied, since
    a zero benchmark is meaningless on a strictly positive series.

    The classification row always carries ``base_rate`` next to
    ``directional_accuracy``. An accuracy without its base rate is unreadable:
    54 percent means skill on a balanced fold and means nothing on a fold that
    was 54 percent up days.
    """
    aligned = y_true.reindex(prediction.index)
    y = np.asarray(aligned.to_numpy(), dtype=float)
    point = np.asarray(prediction.point, dtype=float)

    if task == REGRESSION:
        return {
            "n_obs": float(np.sum(np.isfinite(y) & np.isfinite(point))),
            "rmse": rmse(y, point),
            "mae": mae(y, point),
            "r2_oos": r2_oos(y, point, benchmark),
        }

    if task == CLASSIFICATION:
        # A model that never overrides predict_proba returns None, so fall back
        # to the point forecast for the threshold based metrics and report nan
        # for the ones that genuinely need a probability.
        proba = prediction.proba
        score = point if proba is None else np.asarray(proba, dtype=float)
        row = {
            "n_obs": float(np.sum(np.isfinite(y) & np.isfinite(score))),
            "directional_accuracy": directional_accuracy(y, score),
            "base_rate": base_rate(y),
            "balanced_accuracy": balanced_accuracy(y, score),
            "matthews_corrcoef": matthews_corrcoef(y, score),
        }
        if proba is None:
            row["roc_auc"] = NAN
            row["brier_score"] = NAN
        else:
            row["roc_auc"] = roc_auc(y, score)
            row["brier_score"] = brier_score(y, score)
        return row

    if task == VOLATILITY:
        row = {
            "n_obs": float(np.sum(np.isfinite(y) & np.isfinite(point))),
            "rmse": rmse(y, point),
            "mae": mae(y, point),
        }
        # No zero benchmark R-squared here. Variance is strictly positive, so
        # SSE against a zero forecast is dominated by the mean square rather
        # than the variance, and the ratio comes out comfortably positive for
        # any forecast of roughly the right magnitude, including one that is
        # statistically independent of the outcome. That number would read as
        # skill in the results table when there is none. QLIKE and
        # Mincer-Zarnowitz are the volatility diagnostics; score R-squared only
        # against a benchmark the caller has chosen deliberately.
        if benchmark is not None:
            row["r2_oos"] = r2_oos(y, point, benchmark)
        # The standalone qlike raises on a bad scale so callers cannot ignore
        # it, but a single bad fold must not abort the run, so the row records
        # nan and the run continues.
        try:
            row["qlike"] = qlike(y, point)
        except ValueError:
            row["qlike"] = NAN
        mz = mincer_zarnowitz(y, point)
        row["mz_alpha"] = mz.alpha
        row["mz_beta"] = mz.beta
        row["mz_r2"] = mz.r2
        row["mz_joint_p"] = mz.joint_p
        return row

    raise ValueError(
        f"unknown task {task!r}, expected {REGRESSION!r}, {CLASSIFICATION!r} or {VOLATILITY!r}"
    )


def summarise_folds(rows: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Mean, std, min, max and count of every numeric metric, per group.

    For summary display only. The per fold table is the primary artifact and is
    what belongs in the writeup: a model averaging 54 percent across folds that
    range from 44 to 63 percent is a different object from one that sits at 54
    percent every quarter, and the mean alone cannot tell them apart. Read the
    std and the range here as a prompt to go back to the per fold rows, not as
    a replacement for them.

    ``count`` is per metric and counts non-null values, so a metric that went
    nan on short folds shows a lower count than its neighbours.
    """
    group_by = ["model"] if by is None else list(by)
    if rows.empty:
        return pd.DataFrame()
    missing = [c for c in group_by if c not in rows.columns]
    if missing:
        raise ValueError(f"grouping columns not in results table: {missing}")

    numeric = [
        c
        for c in rows.columns
        if c not in group_by and pd.api.types.is_numeric_dtype(rows[c])
    ]
    if not numeric:
        return pd.DataFrame()

    summary = rows.groupby(group_by, dropna=False)[numeric].agg(
        ["mean", "std", "min", "max", "count"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()
