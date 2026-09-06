"""GARCH and EGARCH conditional volatility models.

This is the family most likely to produce a positive result, and the reason is
structural rather than lucky. Direction is close to unforecastable in a liquid
market because any forecastable component gets traded away. Volatility is not:
volatility clustering survives arbitrage because knowing tomorrow will be
turbulent is not by itself a way to make money without taking risk.

Scale note. The ``arch`` package conditions much better on returns expressed in
percent, so returns are multiplied by 100 before fitting and every forecast
variance is divided by 100 squared on the way out. Forecasts therefore come
back in the same decimal squared units as the Parkinson realised variance they
are scored against, which is what makes the QLIKE comparison meaningful.

Out-of-sample discipline matches the ARIMA module. Parameters are estimated on
the training window and then frozen, and the variance recursion is run forward
over the training and test returns together, so the one step ahead variance at
each test date uses observed returns up to that date and nothing later.

The recursion is driven directly rather than through ``arch``'s ``fix()``.
``fix()`` is convenient, but it derives everything that seeds the recursion
from whatever series it is handed: the backcast, the demeaning constant behind
it, and the variance bounds that clamp it are all computed over train and test
at once. None of it is large enough to move a forecast much, but all of it is a
dependence on data the model is not supposed to have seen. The three quantities
are therefore computed from the training prefix here and handed to the
volatility recursion explicitly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src import schema
from src.models.base import REGRESSION, FoldData, Model
from src.models.arima import _returns_by_asset

# arch fits far more reliably on percent returns than on decimals.
_PERCENT = 100.0

# arch's own seeding window for the backcast and the variance bounds.
_INIT_WINDOW = 75


def _train_only_var_bounds(resids: np.ndarray, n_train: int) -> np.ndarray:
    """``arch``'s variance bounds, with every global constant taken from train.

    The exponentially weighted band down the middle of these bounds is already
    causal: the value at ``t`` is a function of residuals up to ``t - 1``. The
    three clamps around it are not. They use the sample variance and the
    largest squared residual of the whole series, which in the prediction path
    would mean the test window. Recomputing them from the training prefix gives
    bounds of the same shape with no dependence on data the model has not seen.
    """
    from arch.univariate.volatility import ewma_recursion

    nobs = resids.shape[0]
    tau = min(_INIT_WINDOW, n_train)
    weights = 0.94 ** np.arange(tau)
    weights = weights / weights.sum()

    band = np.zeros(nobs, dtype=float)
    ewma_recursion(0.94, resids, band, nobs, float(weights.dot(resids[:tau] ** 2.0)))
    bounds = np.vstack((band / 1e6, band * 1e6)).T

    train = resids[:n_train]
    largest_square = float(np.max(train**2.0))
    lower = float(np.var(train)) / 1e8
    min_upper, upper = 1.0 + largest_square, 1e7 * (1.0 + largest_square)
    bounds[bounds[:, 0] < lower, 0] = lower
    bounds[bounds[:, 1] < min_upper, 1] = min_upper
    bounds[bounds[:, 1] > upper, 1] = upper
    return np.ascontiguousarray(bounds)


class GarchModel(Model):
    """GARCH(1,1) or EGARCH(1,1) forecasting next-day return variance."""

    name = "garch"

    def __init__(self, params: dict | None = None, task: str = REGRESSION, seed: int = 42):
        super().__init__(params, task, seed)
        self.variant = self.params.get("variant", "garch11")
        self.dist = self.params.get("dist", "t")
        self.mean = self.params.get("mean", "constant")
        self.name = self.variant

        # How the model's variance definition is mapped onto the realised
        # measure. "level" shifts by a constant, "mz" fits an intercept and a
        # slope, "none" applies nothing. Both corrections are estimated on the
        # training fold only. Both are reported, because choosing between them
        # by test performance would be precisely the selection this project is
        # built to avoid.
        self.calibration_mode = str(self.params.get("calibration", "level"))
        self.calibrate = self.calibration_mode != "none"
        if self.calibration_mode != "level":
            self.name = f"{self.variant}_{self.calibration_mode}"

        self.fitted_params_: dict[str, pd.Series] = {}
        self.train_series_: dict[str, pd.Series] = {}
        self.fallback_log_var_: dict[str, float] = {}
        self.calibration_: dict[str, tuple[float, float]] = {}
        self.bounds_: dict[str, tuple[float, float]] = {}

    def _is_stable(self, params: pd.Series) -> bool:
        """Reject a parameter set whose variance recursion does not settle."""
        try:
            if self.variant == "egarch11":
                # EGARCH is written in logs, so stability is a condition on the
                # autoregressive term alone.
                beta = float(params.get("beta[1]", 0.0))
                return abs(beta) < 0.999
            alpha = float(params.get("alpha[1]", 0.0))
            beta = float(params.get("beta[1]", 0.0))
            return (alpha + beta) < 0.999 and alpha >= 0.0 and beta >= 0.0
        except Exception:
            return False

    def _make_model(self, series: pd.Series):
        from arch import arch_model

        if self.variant == "egarch11":
            return arch_model(
                series, mean=self.mean, vol="EGARCH", p=1, o=0, q=1, dist=self.dist
            )
        return arch_model(series, mean=self.mean, vol="GARCH", p=1, o=0, q=1, dist=self.dist)

    def _conditional_variance(
        self, params: pd.Series, combined: pd.Series, n_train: int
    ) -> np.ndarray:
        """Variance path over ``combined`` under frozen parameters.

        Everything the recursion needs beyond the parameters themselves comes
        from the first ``n_train`` observations: the mean it is demeaned by is
        the fitted one, the backcast that seeds it reads the training prefix,
        and the bounds that clamp it are the training bounds. The recursion is
        then a pure forward pass, so the variance at ``t`` depends on returns
        strictly before ``t`` and on nothing else.
        """
        volatility = self._make_model(combined).volatility

        # "constant" carries a mu, "zero" does not; both are handled by the
        # default. Reading parameters by name rather than by position keeps
        # GARCH and EGARCH correct without a branch.
        mu = float(params.get("mu", 0.0))
        resids = np.ascontiguousarray(combined.to_numpy(dtype=float) - mu)
        vol_params = np.asarray(
            [float(params[name]) for name in volatility.parameter_names()], dtype=float
        )

        backcast = volatility.backcast(resids[:n_train])
        var_bounds = _train_only_var_bounds(resids, n_train)

        sigma2 = np.zeros(resids.shape[0], dtype=float)
        volatility.compute_variance(vol_params, resids, sigma2, backcast, var_bounds)
        return sigma2

    def fit(self, fold: FoldData) -> "GarchModel":
        for asset, series in _returns_by_asset(fold.meta_train).items():
            clean = series.dropna() * _PERCENT
            if len(clean) < 250:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = self._make_model(clean).fit(disp="off", show_warning=False)
                if not self._is_stable(result.params):
                    # A GARCH whose persistence reaches or exceeds one has no
                    # finite unconditional variance, and iterating its recursion
                    # forward produces a path that drifts without bound. This
                    # happens on the short, extremely volatile early history of
                    # a young asset. Such a fit is discarded rather than used,
                    # because it does not merely forecast badly, it forecasts
                    # backwards: on SOL an unstable fit scored a correlation of
                    # -0.34 against realised variance while the stable fits on
                    # the other three assets scored around +0.5.
                    continue
                self.fitted_params_[asset] = result.params
                self.train_series_[asset] = clean
                # Fallback for any test row the model cannot reach: the training
                # median log variance, which is a constant volatility forecast.
                self.fallback_log_var_[asset] = float(
                    np.log(np.maximum((clean / _PERCENT).var(), 1e-12))
                )
                self.calibration_[asset] = self._estimate_calibration(asset, result, fold)
                self.bounds_[asset] = self._training_bounds(asset, fold)
            except Exception:
                continue

        self.fitted_ = True
        return self

    def _estimate_calibration(self, asset: str, result, fold: FoldData) -> tuple[float, float]:
        """Intercept and slope mapping the model's log variance onto the realised measure.

        Two things need correcting and a single scalar only fixes one of them.

        The level differs because GARCH is estimated on close to close returns
        while the target is a Parkinson range estimator. Parkinson excludes the
        overnight and jump component, so it sits systematically lower, by about
        0.8 in logs on this panel.

        The scale differs too. A conditional variance forecast is much smoother
        than a single day realised estimate, so regressing one on the other
        gives a slope well away from one.

        Fitting both on the training fold is the Mincer-Zarnowitz form of the
        correction. It is train-only, so it carries no test information, and
        with only two degrees of freedom held constant across the whole test
        window it cannot track anything and therefore cannot manufacture skill.
        """
        if not self.calibrate:
            return (0.0, 1.0)
        if "realised_var_now" not in fold.meta_train.columns:
            return (0.0, 1.0)

        try:
            conditional = np.asarray(result.conditional_volatility, dtype=float) ** 2
            model_log_var = np.log(np.maximum(conditional / (_PERCENT**2), 1e-16))

            realised = fold.meta_train.xs(asset, level=schema.ASSET)["realised_var_now"]
            realised_log = np.log(realised.replace(0.0, np.nan))

            n = min(len(model_log_var), len(realised_log))
            if n < 100:
                return (0.0, 1.0)

            x = model_log_var[-n:]
            y = realised_log.to_numpy()[-n:]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 100:
                return (0.0, 1.0)

            if self.calibration_mode == "level":
                # The minimal correction for a known definitional gap: one
                # constant, no fitted slope, nothing that can drift.
                return (float(np.mean(y[ok] - x[ok])), 1.0)

            slope, intercept = np.polyfit(x[ok], y[ok], 1)
            # A non-positive or wild slope means the regression has not found a
            # usable relationship, so fall back to a pure level shift.
            if not (0.05 < slope < 5.0):
                return (float(np.mean(y[ok] - x[ok])), 1.0)
            return (float(intercept), float(slope))
        except Exception:
            return (0.0, 1.0)

    def _training_bounds(self, asset: str, fold: FoldData) -> tuple[float, float]:
        """Plausible range for a log variance forecast, taken from train."""
        try:
            realised = fold.meta_train.xs(asset, level=schema.ASSET)["realised_var_now"]
            log_var = np.log(realised.replace(0.0, np.nan).dropna())
            centre, spread = float(log_var.mean()), float(log_var.std())
            return centre - 5.0 * spread, centre + 5.0 * spread
        except Exception:
            return (-np.inf, np.inf)

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        """Log of the one step ahead conditional variance, in decimal squared units."""
        out = pd.Series(np.nan, index=X.index, dtype=float)
        if meta is None:
            return np.zeros(len(X))

        for asset, series in _returns_by_asset(meta).items():
            params = self.fitted_params_.get(asset)
            fallback = self.fallback_log_var_.get(asset, np.log(1e-4))

            if params is None:
                out.loc[asset] = fallback
                continue
            intercept, slope = self.calibration_.get(asset, (0.0, 1.0))
            fallback = intercept + slope * fallback

            test_series = series.ffill().fillna(0.0) * _PERCENT
            if test_series.empty:
                continue
            train_series = self.train_series_.get(asset)
            combined = (
                pd.concat([train_series, test_series])
                if train_series is not None
                else test_series
            )
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()

            # Where the training prefix ends. Strictly before the first test
            # date, so a duplicated boundary row resolved in favour of the test
            # value cannot be counted as training data. The guard below is
            # unreachable in practice, since fit() refuses an asset with fewer
            # than 250 training returns, but the recursion cannot be seeded
            # from a shorter prefix than arch's own window and saying so is
            # cheaper than assuming the two constants stay in that order.
            n_train = int((combined.index < test_series.index[0]).sum())
            if n_train < _INIT_WINDOW:
                out.loc[asset] = fallback
                continue

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    conditional = pd.Series(
                        self._conditional_variance(params, combined, n_train),
                        index=combined.index,
                    )

                # Read the recursion directly rather than calling forecast().
                # In a GARCH model, sigma squared at t+1 is a function of the
                # squared innovation at t and sigma squared at t, so the value
                # the recursion holds for t+1 is already known at t. Taking the
                # next row is therefore a one step ahead forecast and not a peek
                # at the future, and it is far easier to verify than the
                # forecast() path, which silently returns a value only at the
                # final observation unless a start date is supplied.
                one_step_ahead = conditional.shift(-1).reindex(test_series.index)

                decimal_variance = one_step_ahead / (_PERCENT**2)
                raw = np.log(decimal_variance.clip(lower=1e-12))
                values = intercept + slope * raw
                # Numerical safety net. Bounds come from the training fold, so
                # they carry no test information, and they only ever bind on a
                # forecast that has already left the plausible range.
                low, high = self.bounds_.get(asset, (-np.inf, np.inf))
                values = values.clip(lower=low, upper=high)
                out.loc[asset] = values.fillna(fallback).to_numpy(dtype=float)
            except Exception:
                out.loc[asset] = fallback

        return out.fillna(np.log(1e-4)).to_numpy(dtype=float)

    def conditional_volatility(self, X: pd.DataFrame, meta: pd.DataFrame | None) -> pd.Series:
        """Forecast standard deviation per period, for volatility-scaled sizing.

        The backtest wants a standard deviation of returns, not a log variance,
        so this converts once in a single place rather than at every call site.
        """
        log_variance = self.predict(X, meta)
        return pd.Series(np.sqrt(np.exp(log_variance)), index=X.index)

    def fitted_parameters(self) -> pd.DataFrame:
        """Estimated parameters per asset, for the report."""
        if not self.fitted_params_:
            return pd.DataFrame()
        return pd.DataFrame(self.fitted_params_).T


def build_garch_models(config: dict, task: str = REGRESSION) -> list[GarchModel]:
    """Instantiate each configured GARCH variant."""
    model_cfg = config.get("model", {})
    variants = model_cfg.get("variants", ["garch11"])
    calibrations = model_cfg.get("calibrations", ["level"])
    return [
        GarchModel(
            params={
                "variant": variant,
                "dist": model_cfg.get("dist", "t"),
                "mean": model_cfg.get("mean", "constant"),
                "calibration": calibration,
            },
            task=task,
        )
        for variant in variants
        for calibration in calibrations
    ]
