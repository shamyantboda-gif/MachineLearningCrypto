"""ARIMA on daily log returns.

Fitted per asset and refitted at every fold boundary. ``d`` is fixed at zero
because the target is already a return series, and differencing a return would
be differencing twice.

Out-of-sample forecasting is done the honest way. Parameters are estimated on
the training window and then frozen. The fitted state is extended through the
test period with ``append(refit=False)``, which feeds in the actual observed
returns one at a time and reads off the one step ahead prediction at each
point. Every forecast for date ``t+1`` therefore uses real data up to ``t`` and
parameters that never saw the test window.

The expected outcome is worth stating in advance: on daily crypto returns the
information criteria usually select an order close to (0, 0, 0), and the model
predicts approximately the training mean. That is not a failure. It is the
model correctly reporting that there is no linear autocorrelation to exploit,
and it is the reason the baselines are so hard to beat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from src import schema
from src.models.base import CLASSIFICATION, FoldData, Model

RET_LAG1 = "ret_lag1"


class ArimaModel(Model):
    """Per-asset ARIMA with order selection by information criterion."""

    name = "arima"
    wants_raw_features = True

    def __init__(self, params: dict | None = None, task: str = CLASSIFICATION, seed: int = 42):
        super().__init__(params, task, seed)
        self.p_range = self.params.get("p_range", [0, 1, 2])
        self.q_range = self.params.get("q_range", [0, 1, 2])
        self.d = int(self.params.get("d", 0))
        self.ic = self.params.get("ic", "aic")
        self.max_train_obs = int(self.params.get("max_train_obs", 2000))

        self.orders_: dict[str, tuple[int, int, int]] = {}
        self.results_: dict[str, object] = {}
        self.sigma_: dict[str, float] = {}
        self.train_mean_: dict[str, float] = {}

    def fit(self, fold: FoldData) -> "ArimaModel":
        from statsmodels.tsa.arima.model import ARIMA

        series_by_asset = _returns_by_asset(fold.meta_train)

        for asset, series in series_by_asset.items():
            series = series.dropna()
            if len(series) < 100:
                continue
            # Long histories add little to a low order model and cost real time,
            # so the fit uses the most recent window.
            series = series.iloc[-self.max_train_obs :]

            best_order, best_result, best_score = None, None, np.inf
            for p in self.p_range:
                for q in self.q_range:
                    if p == 0 and q == 0:
                        # A pure mean model is a legitimate and common choice
                        # here, so it stays in the grid rather than being
                        # excluded as trivial.
                        pass
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted = ARIMA(
                                series,
                                order=(p, self.d, q),
                                trend="c",
                                enforce_stationarity=True,
                                enforce_invertibility=True,
                            ).fit()
                        score = fitted.aic if self.ic == "aic" else fitted.bic
                    except Exception:
                        continue
                    if np.isfinite(score) and score < best_score:
                        best_order, best_result, best_score = (p, self.d, q), fitted, score

            if best_result is None:
                continue

            self.orders_[asset] = best_order
            self.results_[asset] = best_result
            residuals = np.asarray(best_result.resid, dtype=float)
            self.sigma_[asset] = float(np.nanstd(residuals)) or 1e-6
            self.train_mean_[asset] = float(series.mean())

        self.fitted_ = True
        return self

    def _point_forecasts(self, X: pd.DataFrame, meta: pd.DataFrame | None) -> pd.Series:
        """One step ahead forecasts for every row, assembled per asset."""
        out = pd.Series(np.nan, index=X.index, dtype=float)
        if meta is None:
            return out.fillna(0.0)

        for asset, series in _returns_by_asset(meta).items():
            result = self.results_.get(asset)
            if result is None:
                fallback = self.train_mean_.get(asset, 0.0)
                out.loc[asset] = fallback
                continue

            clean = series.ffill().fillna(0.0)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # refit=False is the whole point: the parameters stay as
                    # estimated on train while the state absorbs real data.
                    extended = result.append(clean, refit=False)
                    predictions = extended.predict(
                        start=len(extended.model.endog) - len(clean),
                        end=len(extended.model.endog) - 1,
                    )
                values = np.asarray(predictions, dtype=float)
            except Exception:
                values = np.full(len(clean), self.train_mean_.get(asset, 0.0))

            out.loc[asset] = values

        return out.fillna(0.0)

    def predict(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        forecasts = self._point_forecasts(X, meta)
        if self.task == CLASSIFICATION:
            return (forecasts.to_numpy() > 0).astype(float)
        return forecasts.to_numpy()

    def predict_proba(self, X: pd.DataFrame, meta: pd.DataFrame | None = None) -> np.ndarray:
        """Turn a point forecast into P(up) using the fitted residual scale.

        Under the model's own assumption of normal errors with standard
        deviation sigma, P(return > 0) is the normal CDF of the forecast
        divided by sigma. That is a real probability rather than a hard label,
        which is what the calibration curve and the backtest both need.
        """
        forecasts = self._point_forecasts(X, meta)
        assets = X.index.get_level_values(schema.ASSET)
        sigma = np.array([self.sigma_.get(a, 1e-2) for a in assets], dtype=float)
        sigma = np.where(sigma > 0, sigma, 1e-6)
        return norm.cdf(forecasts.to_numpy() / sigma)

    def selected_orders(self) -> pd.DataFrame:
        """Order chosen per asset, reported because (0,0,0) is itself a finding."""
        return pd.DataFrame(
            [{"asset": a, "order": str(o)} for a, o in sorted(self.orders_.items())]
        )


def _returns_by_asset(meta: pd.DataFrame) -> dict[str, pd.Series]:
    """Split the lagged return column into one clean series per asset.

    ``ret_lag1`` at date ``t`` is the return realised over ``t-1`` to ``t``, so
    a series of it indexed by date is exactly the observed return process.
    """
    if RET_LAG1 not in meta.columns:
        return {}
    out: dict[str, pd.Series] = {}
    for asset, group in meta.groupby(level=schema.ASSET, observed=True):
        series = group[RET_LAG1].droplevel(schema.ASSET).sort_index()
        series.index = pd.DatetimeIndex(series.index)
        out[str(asset)] = series.astype(float)
    return out


def stationarity_tests(returns: pd.Series) -> dict[str, float]:
    """ADF and KPSS on a return series.

    Returns are almost always stationary and prices almost never are, which is
    the whole reason this project models returns. Running the tests and
    reporting the numbers is cheap and makes the claim checkable.
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    clean = returns.dropna()
    out: dict[str, float] = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_stat, adf_p = adfuller(clean, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(clean, regression="c", nlags="auto")[:2]

    out["adf_stat"] = float(adf_stat)
    out["adf_pvalue"] = float(adf_p)
    out["kpss_stat"] = float(kpss_stat)
    out["kpss_pvalue"] = float(kpss_p)
    return out
