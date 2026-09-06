"""Diebold-Mariano test for equal predictive accuracy of two forecasts.

The test asks whether two forecasts of the same realised series differ in
expected loss. It does not ask whether either forecast is any good. Pairing it
with the baseline comparison in :func:`dm_vs_baseline` is what turns "model A
has a lower RMSE this fold" into a claim with a p-value attached.

Sign convention, stated once and repeated on every public function because it
is the thing readers get backwards: the loss differential is

    d_t = L(e_a_t) - L(e_b_t)

so a NEGATIVE mean differential means model A carries the lower loss, which
means A is the better forecast. A negative statistic with a small p-value is
evidence for A. A positive statistic with a small p-value is evidence for B.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

VALID_LOSSES = ("squared", "absolute", "qlike")


@dataclass(frozen=True)
class DMResult:
    """Outcome of one Diebold-Mariano comparison.

    ``statistic`` and ``p_value`` are nan on a degenerate comparison, in which
    case ``note`` says why. ``better`` reads off the sign of
    ``mean_loss_diff``: negative means "a", positive means "b". It is not
    gated on significance, so it carries the plain meaning "this one had the
    lower loss on this sample" and says nothing about whether the gap is
    distinguishable from noise. Read it next to ``p_value``, never alone.
    """

    statistic: float
    p_value: float
    mean_loss_diff: float
    n_obs: int
    loss: str
    better: str
    note: str = ""

    def __str__(self) -> str:
        if not np.isfinite(self.statistic):
            reason = self.note or "degenerate comparison"
            return f"DM[{self.loss}] n={self.n_obs} not computed: {reason}"
        return (
            f"DM[{self.loss}] stat={self.statistic:+.3f} p={self.p_value:.4f} "
            f"mean_diff={self.mean_loss_diff:+.6g} n={self.n_obs} better={self.better}"
        )


def _loss_series(y: np.ndarray, f: np.ndarray, loss: str) -> np.ndarray:
    """Per observation loss of forecast ``f`` against realisation ``y``."""
    if loss == "squared":
        return (y - f) ** 2
    if loss == "absolute":
        return np.abs(y - f)
    if loss == "qlike":
        # QLIKE lives on the variance scale and is undefined for a ratio that
        # is zero or negative, so this is a hard error rather than a nan. A
        # caller silently scoring log variances would get a plausible looking
        # number that means nothing.
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
        return ratio - np.log(ratio) - 1.0
    raise ValueError(f"unknown loss {loss!r}, expected one of {VALID_LOSSES}")


def _degenerate(mean_diff: float, n_obs: int, loss: str, note: str) -> DMResult:
    """Build a not-computed result.

    ``better`` is always "tie" on these paths, so a row with a nan statistic
    never reads as a winner. The raw ``mean_loss_diff`` is still reported for
    anyone who wants to look at the sample gap.
    """
    return DMResult(
        statistic=float("nan"),
        p_value=float("nan"),
        mean_loss_diff=mean_diff,
        n_obs=n_obs,
        loss=loss,
        better="tie",
        note=note,
    )


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    horizon: int = 1,
    loss: str = "squared",
    harvey_correction: bool = True,
) -> DMResult:
    """Test the null of equal expected loss for two forecasts of ``y_true``.

    Sign convention: the loss differential is ``d_t = L(e_a_t) - L(e_b_t)``.
    A negative ``mean_loss_diff`` and a negative ``statistic`` mean model A has
    the lower loss, so A is better. Positive means B is better.

    Parameters
    ----------
    y_true:
        Realised series. For ``loss="qlike"`` this must be realised VARIANCE,
        strictly positive. Not a log variance and not a standard deviation.
    pred_a, pred_b:
        The two competing forecasts, on the same scale as ``y_true``.
    horizon:
        Forecast horizon h. The long run variance truncates at ``h - 1`` lags,
        so ``h=1`` uses the sample variance alone and needs no lag terms.
    loss:
        One of ``"squared"``, ``"absolute"``, ``"qlike"``.
    harvey_correction:
        Apply the Harvey, Leybourne and Newbold small sample correction and
        take the p-value from Student-t with ``n - 1`` degrees of freedom.
        Without it the statistic is referred to the standard normal, which
        over rejects on the short test windows a walk-forward split produces.

    Returns
    -------
    DMResult
        On a degenerate comparison (identical forecasts, fewer than five
        observations, a constant loss differential, or a non-positive long run
        variance) the statistic and p-value are nan and ``note`` explains why.
        Returning nan rather than raising is deliberate: a single short or tied
        fold must not abort a walk-forward run that has twenty other folds to
        report.
    """
    if loss not in VALID_LOSSES:
        raise ValueError(f"unknown loss {loss!r}, expected one of {VALID_LOSSES}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    y = np.asarray(y_true, dtype=float).ravel()
    fa = np.asarray(pred_a, dtype=float).ravel()
    fb = np.asarray(pred_b, dtype=float).ravel()
    if not (y.shape == fa.shape == fb.shape):
        raise ValueError(
            f"length mismatch: y_true={y.shape}, pred_a={fa.shape}, pred_b={fb.shape}"
        )

    # Drop rows where any of the three is missing, jointly, so the two loss
    # series are always computed on the same sample. Without this one stray nan
    # propagates into the mean and the result looks like a degenerate fold
    # rather than a data problem.
    finite = np.isfinite(y) & np.isfinite(fa) & np.isfinite(fb)
    n_dropped = int((~finite).sum())
    y, fa, fb = y[finite], fa[finite], fb[finite]
    n = int(y.size)
    drop_note = f" ({n_dropped} non-finite rows dropped)" if n_dropped else ""

    if n == 0:
        return _degenerate(float("nan"), 0, loss, "no finite observations" + drop_note)

    # The loss call is what enforces qlike positivity, so it runs before the
    # degenerate checks: bad input should raise, not be reported as a tie.
    loss_a = _loss_series(y, fa, loss)
    loss_b = _loss_series(y, fb, loss)
    d = loss_a - loss_b
    mean_diff = float(np.mean(d))

    if n < 5:
        return _degenerate(
            mean_diff, n, loss, f"only {n} observations, need at least 5" + drop_note
        )
    if np.array_equal(fa, fb):
        return _degenerate(
            mean_diff, n, loss, "forecasts are identical" + drop_note
        )

    better = "tie"
    if mean_diff < 0:
        better = "a"
    elif mean_diff > 0:
        better = "b"

    dc = d - mean_diff
    # gamma_0 uses divisor n, not n - 1. With the Harvey factor at h=1 this
    # makes the statistic exactly the one sample t statistic on d, which is
    # what gives the test its nominal size. Using n - 1 here would make it
    # quietly conservative.
    gamma_0 = float(dc @ dc) / n
    if gamma_0 <= 0.0:
        return _degenerate(
            mean_diff, n, loss, "loss differential is constant" + drop_note
        )

    long_run = gamma_0
    max_lag = min(horizon - 1, n - 1)
    for k in range(1, max_lag + 1):
        gamma_k = float(dc[k:] @ dc[:-k]) / n
        long_run += 2.0 * gamma_k

    # Unweighted truncation at h - 1 lags is the original DM recipe but it is
    # not a positive semi-definite estimator, so at h > 1 it can come out
    # negative. That is an estimator failure, not a result.
    if long_run <= 0.0:
        return _degenerate(
            mean_diff,
            n,
            loss,
            f"long run variance estimate is non-positive at horizon {horizon}" + drop_note,
        )

    avar = long_run / n
    statistic = mean_diff / np.sqrt(avar)

    if harvey_correction:
        h = float(horizon)
        factor = (n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n
        if factor <= 0.0:
            return _degenerate(
                mean_diff,
                n,
                loss,
                f"Harvey correction undefined: horizon {horizon} too large for n={n}"
                + drop_note,
            )
        statistic *= np.sqrt(factor)
        p_value = 2.0 * float(stats.t.sf(abs(statistic), df=n - 1))
    else:
        p_value = 2.0 * float(stats.norm.sf(abs(statistic)))

    return DMResult(
        statistic=float(statistic),
        p_value=p_value,
        mean_loss_diff=mean_diff,
        n_obs=n,
        loss=loss,
        better=better,
        note=drop_note.strip(),
    )


def dm_vs_baseline(
    y_true: np.ndarray,
    pred_model: np.ndarray,
    pred_baseline: np.ndarray,
    **kwargs,
) -> DMResult:
    """Compare a model against a baseline, with the model in the A slot.

    A negative statistic with a small p-value means the model genuinely beats
    the baseline: lower loss, and a gap large enough that it is unlikely to be
    sampling noise. A positive statistic with a small p-value means the
    baseline wins. A large p-value in either direction means this fold cannot
    tell the two apart, which on daily crypto direction is the expected result.

    Keyword arguments pass straight through to :func:`diebold_mariano`.
    """
    return diebold_mariano(y_true, pred_model, pred_baseline, **kwargs)
