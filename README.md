# Forecasting daily cryptocurrency returns

Seven model families against four naive baselines on daily Binance bars for BTC, ETH, LTC and SOL, evaluated walk-forward over 27 quarterly folds (2020-01 to 2026-07, 9,299 scored predictions per model) with purging, embargo, per-fold scaling, and a Diebold-Mariano test on every comparison.

**Result: nothing beat abstaining.** The best model (LightGBM) called next-day direction 51.35% of the time against a 51.12% base rate, a 0.23-point edge inside a 3.6-point fold-to-fold spread (DM p = 0.97). Reseeding the LSTM moves its accuracy 5× more than the effect being measured. Traded long-short, the LightGBM signal earned 8.6%/yr at zero cost and -21.4% at realistic taker fees; buy-and-hold returned 56.2%. GARCH(1,1) lost to a 63-day trailing mean on volatility because its calibration drifts fold to fold. The one thing the study cannot say is that *no* edge exists: 27 folds can't detect a half-point one.

---

## Direction: next day, 27 folds, 9,299 predictions per model

| model | accuracy | base rate | ROC-AUC | MCC | folds beating base rate | DM p vs abstain |
|---|---|---|---|---|---|---|
| lightgbm | 0.5135 (SD 0.0357) | 0.5112 | 0.5298 | 0.0414 | 16 of 27 | 0.97 |
| ridge / L2 logistic | 0.5115 (SD 0.0418) | 0.5112 | 0.5315 | 0.0397 | 14 of 27 | 0.001 |
| zero (abstain at 0.5) | 0.5112 | 0.5112 | 0.5000 | 0.0000 | n/a | n/a |
| historical mean | 0.5044 | 0.5112 | 0.4548 | -0.0425 | 7 of 27 | <0.001 |
| majority class | 0.4875 | 0.5112 | 0.5000 | 0.0000 | 0 of 27 | <0.001 |
| arima | 0.4833 | 0.5112 | 0.4936 | -0.0080 | 4 of 27 | <0.001 |
| persistence | 0.4771 | 0.5112 | 0.4630 | -0.0539 | 7 of 27 | <0.001 |

The DM column is a Diebold-Mariano test on Brier scores against the abstaining forecast, with the Harvey-Leybourne-Newbold small-sample correction. Every model except LightGBM is significantly *worse* than a constant 0.5, and LightGBM does not beat it either.

**Ranking is not calibration.** Ridge posted the best ROC-AUC in the study (0.5315) and still loses to abstaining at p = 0.001. AUC asks whether up-days are ranked above down-days; Brier asks whether the printed number is a probability. Ridge's per-fold Brier ranges 0.240 to 0.285 against the abstainer's 0.250 by construction: it is confident at the wrong times. Reporting AUC alone would have hidden that. Whatever weak structure exists in these 80 features is linear, and the boosted trees find nothing on top of it.

**Persistence at 47.7% is a real effect, not noise.** Daily crypto returns mean-revert slightly at a one-day horizon, so betting that tomorrow repeats today loses more often than a coin flip.

### Sequence models, 5 seeds each, same folds and rows

| model | accuracy | seed SD within fold | ROC-AUC | MCC | folds beating base rate |
|---|---|---|---|---|---|
| lstm | 0.5060 (SD 0.0352) | 0.0261 | 0.5205 | 0.0225 | 12 of 27 |
| dlinear | 0.5048 (SD 0.0329) | 0.0228 | 0.5134 | 0.0166 | 13 of 27 |
| cnn | 0.4968 (SD 0.0269) | 0.0181 | 0.5044 | -0.0012 | 10 of 27 |

None reached the base rate. DM against abstaining: DLinear and CNN significantly worse (p < 0.001), LSTM indistinguishable (p = 0.20). The LSTM's mean edge over abstaining is -0.53 points against a within-fold seed SD of 2.61 points, so a single-seed run could report almost anything. DLinear, a single linear layer over a decomposed input, finished within 0.12 points of the LSTM in a fifth of the training time (3.6s per fit vs 7.6s LSTM, 13.5s CNN). That is the Zeng et al. point reproduced here: across all seven families the ordering is close to the reverse of their complexity.

### Slices

Per asset, LightGBM minus that asset's own base rate: BTC +0.70pp, LTC +0.66pp, SOL +0.40pp, ETH -0.43pp. If less liquid markets were more predictable, SOL should lead and BTC should trail; neither happens. SOL is also measured on a different window (listed August 2020, absent from folds 0–2, 2,090 scored rows vs 2,403 for the others), so its number is the weakest evidence in the row.

By regime: +0.42pp bear, +0.46pp sideways, +0.04pp bull. All inside the fold-to-fold spread.

Feature importance is not stable enough to interpret: `xa_ref_ret_lag1` ranks 2nd on one fold and 53rd on another. By mean gain the top features were `px_ret_1`, `rng_close_position`, `xa_market_ret_lag1` and `xa_market_dispersion`; six of the top twelve are cross-asset, the one place the design expected to find something.

## Volatility: next-day log realised variance, 27 folds

| model | QLIKE | MZ slope | MZ R2 | RMSE (log variance) |
|---|---|---|---|---|
| trailing mean, 63d | 1.009 | 1.034 | 0.056 | 1.086 |
| persistence | 1.097 | 0.323 | 0.128 | 1.164 |
| EGARCH(1,1) | 5.709 | 0.555 | 0.082 | 1.388 |
| GARCH(1,1) | 7.929 | 0.558 | 0.065 | 1.688 |

GARCH is not short of information: per asset its forecasts correlate 0.46–0.53 with next-day realised variance, about the same as persistence. Its level is wrong. Per-fold bias has a standard deviation of 1.34 in log variance against 0.02 for persistence, so the ranking never gets to pay off. Two corrections fitted on training folds only (constant shift; intercept-plus-slope) landed within 1% of each other, so this is not an artifact a better correction would remove. The Mincer-Zarnowitz columns show the split: persistence tracks variation best (highest R2) but is badly scaled (slope 0.32); the trailing mean is almost perfectly calibrated (slope 1.03) and wins by being unbiased rather than informative.

## Backtest: long-short, 2% dead zone

The rule is per asset, on the predicted probability of an up move: long at `p > 0.52`, short at `p < 0.48`, flat in between. That 0.02 band is the dead zone, and it is what stops a model with no view from trading on rounding noise. A missing probability is treated as no view rather than as 0.5. Positions are unit sized, then equal weighted across whichever assets hold one that day, so each date's weights sum to one in absolute terms and a four-asset run stays on the same scale as a one-asset run. Turnover is charged per asset as the change in position, starting from a flat book, so the opening trade is paid for. Volatility targeting exists in the engine but is off for these results.

| signal | cost (bps round trip) | annual return | Sharpe | max drawdown | annual turnover |
|---|---|---|---|---|---|
| lightgbm | 0 | 8.59% | 0.44 | -77.9% | 323 |
| lightgbm | 5 | 0.16% | 0.31 | -81.4% | 323 |
| lightgbm | 20 | -21.42% | -0.09 | -92.3% | 323 |
| ridge | 0 | -1.85% | 0.34 | -85.5% | 416 |
| ridge | 5 | -11.55% | 0.19 | -88.6% | 416 |
| ridge | 20 | -35.30% | -0.25 | -95.7% | 416 |
| buy and hold | n/a | 56.16% | 0.98 | -83.0% | 0.23 |

Binance spot taker fees are around 10 bps one way, so the 20 bps row is the realistic one.

Ridge is the sharper test and fails earlier: it loses 1.85%/yr before any cost while trading 29% more than LightGBM. The best-ranking model is the worse strategy at every cost level. The problem is not that the edge is small; ranking skill this weak does not survive being turned into positions.

Sharpe and annual return disagree on the ridge rows because Sharpe uses the arithmetic mean of daily returns and annual return is compounded. Ridge at zero cost has an arithmetic mean of +24.0%/yr and volatility of 71.2%, so variance drag of roughly half the squared volatility (25.3%) exceeds the mean. The compounded number is what an account would experience, so it is the one to read.

## Data

Daily klines from the Binance public archive (`data.binance.vision`): a static host, no key, no rate limit, SHA256 checksum on every monthly zip. `src/data/fetch_binance.py` verifies all of them. No price data is committed; `make data` rebuilds it.

| asset | symbol | first bar | bars | first bar scored | bars scored | enters at fold |
|---|---|---|---|---|---|---|
| Bitcoin | BTCUSDT | 2017-08-17 | 3,271 | 2020-01-01 | 2,403 | 0 |
| Ethereum | ETHUSDT | 2017-08-17 | 3,271 | 2020-01-01 | 2,403 | 0 |
| Litecoin | LTCUSDT | 2017-12-13 | 3,153 | 2020-01-01 | 2,403 | 0 |
| Solana | SOLUSDT | 2020-08-11 | 2,181 | 2020-11-09 | 2,090 | 3 |

11,876 rows, no gaps, no duplicate timestamps, no OHLC ordering violations, no zero-volume bars. A bar is scored only inside a test window (first opens 2020-01-01) and at least 90 bars after its own listing, since `px_cumret_90` is the longest feature window. Data is held long (one row per asset-date) so SOL can start when it started without dropping or inventing rows for the other three.

**Cross-check.** BTC closes against Coinbase: 95.2% of 3,271 overlapping days agree within 1%. Every disagreement is in 2017–2019 (49 of 137 days in 2017, 71 in 2018, 35 in 2019, 2 in 2020, 0 of 2,038 from 2021 on). Binance quotes USDT and Coinbase quotes USD; the venues converged once arbitrage in the pair became routine. The disputed region sits almost entirely inside training warmup.

Three daily log returns exceed 50% in absolute value: ETH -0.59 and BTC -0.50 on 2020-03-12 (COVID crash), SOL -0.55 on 2022-11-09 (FTX). Real events, kept.

Two things to know if you write your own fetcher: the Binance archive switched from millisecond to microsecond timestamps mid-history, so the unit has to be detected per row, not per file; and prices are never forward-filled here, because a filled price is a zero return that drags every volatility estimate down. Coin Metrics community tier covers BTC, ETH and LTC but not SOL, so on-chain features are missing for one asset. `reports/data_quality.md` regenerates all of this on every build.

## Features

80 features in 7 toggleable families: price (25: log returns at 6 lags, rolling moments, drawdown position), cross-asset (13: lagged BTC return, rolling correlation to BTC, cross-sectional rank), range (9: Parkinson, Garman-Klass, ATR, close position in bar), volume (9: quote volume z-score, taker buy ratio, trade count z-score), auxiliary (9: Fear and Greed, on-chain z-scores), technical (8: RSI, MACD, Bollinger position, stochastic, OBV z-score), calendar (7: sine/cosine day and month). The technical indicators are hand-rolled and add almost nothing: they are deterministic transforms of the same lagged prices the price family already holds, contributing nonlinearity rather than information.

## Design choices that separate a result from an artifact

**Returns, not prices.** Predicting tomorrow's price scores R² ≈ 0.99 by outputting today's. Every target in `src/data/targets.py` is a return, a direction, or a log variance.

**Walk-forward, never shuffled.** `src/splits/walk_forward.py`: train from 2018-01, test one quarter, step one quarter, 27 folds. Training always ends before testing begins.

**Scaler fitted inside the fold.** `src/preprocess.py` builds a fresh scaler per fold and never sees a test row. Standardising over the whole dataset leaks future volatility backward, does not crash, and just makes the numbers better than they should be.

**Purging and embargo are different things.** Purging removes training rows whose *label* overlaps the test period: one day, at a one-day horizon. Feature windows never need purging however long they are, because a 63-day mean ending at `t` contains nothing after `t`. Embargo drops a further 5 days so no training row sits directly against the boundary where serial correlation makes it nearly the same observation. Required gap 6 days; measured gap 7, because the splitter keeps rows strictly before the boundary. `tests/test_no_leakage.py` asserts this at several embargo settings, including zero.

## Bugs found while building this

Most were caught by measurement rather than by reading code, which is the argument for building the evaluation harness first. The last was not, which is the argument for reading it anyway.

- **GARCH forecasting a constant.** `arch`'s `forecast()` returns a value only at the final observation unless passed `start`; every other test row silently fell back to a per-asset constant. Symptom: plausibly scaled forecasts correlating -0.05 with realised variance.
- **Degenerate GARCH fit on SOL.** Alpha plus beta above 1 on its short, violent early history (no finite unconditional variance), scoring -0.34 correlation against ~+0.5 elsewhere and dragging the pooled number down. Now a stationarity guard in `src/models/garch.py`.
- **Diebold-Mariano comparing labels to probabilities.** Hard 0/1 labels against a constant 0.5 under squared loss made abstaining unbeatable by construction. It was measuring output encoding.
- **A volatility baseline in the wrong units.** The historical-mean baseline predicted a mean return of ~0.001 against a log-variance target of ~-7, making everything else look good.
- **Rolling windows spanning two assets.** A missing `groupby(level='asset')`. Truncating one asset's history can't catch it because the other asset's rows are untouched; comparing a single-asset build against the same asset's rows in a multi-asset build does.
- **GARCH reading its own future through `arch`.** The one-step recursion is causal, but `fix()` derives its backcast, demeaning constant and variance bounds from the whole series handed to it, which included the test window. Multiplying the tail of a test window by twenty moved the variance at the start of the recursion by ~2%; the influence decays geometrically in beta and is gone by the test rows. Re-running after the fix left 96% of predictions bit-identical and moved every metric by no more than 2e-15. Still a dependence on future data. `src/models/garch.py` now computes those three quantities from the training prefix and drives the recursion itself; `tests/test_no_leakage.py` perturbs the tail of a test window and requires every earlier forecast to come back bit-identical.

## Limitations

- **27 folds is not many.** Fold-to-fold SD of accuracy is ~3.6 points. The honest reading of the null is that no *large* edge exists, not that none does.
- **No multiple-comparisons correction.** Seven models, four assets, three regimes and two targets are compared and none survive, so the direction of the omission is conservative; it would not be if something had.
- **Survivorship.** The four assets were chosen for long clean histories, i.e. because they survived. Conservative for a null result, not for a positive one.
- **Thin, uneven cross-section.** Four heavily correlated assets, three of them in the first three folds. Any asset-to-asset comparison is also a window-to-window comparison.
- **One horizon.** Everything is one day ahead. Skill decay with horizon is a real result this project does not have.
- **USDT, not USD.** Deepest books and longest history, but the peg has broken briefly on a handful of days.
- **One hand-argued exception in the leakage scan.** `src/models/garch.py` is the only file under `src/` allowed a backward shift, because reading the variance recursion one row ahead is a genuine one-step forecast. The justification is written next to the allowlist entry in `tests/test_no_leakage.py` and backed by two model-specific tests. Any new negative shift fails the suite until someone justifies it the same way.

## Not done, named rather than omitted

Hourly bars (same archive, 24× the data, would change the power of every test above). Point-in-time universe from the full set of USDT pairs including delisted ones. Horizon-decay curve. Order-book imbalance from the depth snapshots in the same archive. Triple-barrier labels, conformal intervals, regime-switching models. Gramian Angular Field encodings for a 2D CNN. Optuna and MLflow: hyperparameters are frozen in YAML and runs are tracked by config hash in a CSV, which is worse than real tracking and better than nothing.

## Reproduce

```bash
pip install -r requirements.txt
make data      # 392 monthly files, checksum-verified, ~15 min first run, cached after
make test      # 84 tests, mostly leakage and alignment
make train     # baselines + ARIMA + ridge + LightGBM on direction
make vol       # baselines + GARCH/EGARCH on volatility
make deep      # LSTM, 1D CNN, DLinear, 5 seeds each (~56 min, CPU)
make backtest  # cost sweep and equity curves
make report    # summary tables and figures
```

Python 3.11+; developed on 3.14.3 with pandas 3.0.5, numpy 2.5.2, torch 2.14.0, all pinned. No GPU anywhere. Notebooks need `jupyter` and `ipykernel`, which are not in `requirements.txt` because nothing in the pipeline imports them.

Every experiment is a YAML file, and the SHA256 of the merged config names its output directory under `reports/results/`. Those directories are committed, predictions included, so any number above can be checked without retraining.

| table | run | command |
|---|---|---|
| direction: ridge, LightGBM, ARIMA | `a08f75962a` | `make train` |
| direction: sequence models | `a2e2e68684` | `src.train --model lstm.yaml --model cnn.yaml --model dlinear.yaml` |
| volatility | `96037aadcc` | `src.train --target vol_1d --model garch.yaml` |
| backtest | `a08f75962a` | `src.run_backtest --run reports/results/a08f75962a` |

Every model in a table was evaluated on the same folds and rows as the baselines beside it. `dm.csv` in each direction run holds the Diebold-Mariano results; the sequence-model directory predates that and was backfilled from its own stored predictions with the same function. `--dm-baseline` changes the comparison without changing the config hash, so reporting a different baseline does not relocate the results.

```
config/            base.yaml plus one file per model family
src/
  data/            fetchers, validation suite, panel builder, targets
  features/        one module per family, plus the registry
  splits/          purged and embargoed walk-forward splitter
  models/          baselines, ARIMA, GARCH, ridge, LightGBM, LSTM, CNN, DLinear
  evaluate/        metrics, Diebold-Mariano, reporting
  backtest/        cost model and engine, written from scratch
  train.py         experiment runner
tests/             alignment, leakage and causality checks
notebooks/         exploration and results analysis, not the pipeline
```

## References

López de Prado, *Advances in Financial Machine Learning*, ch. 7 (purging and embargo). Zeng et al. (2022), "Are Transformers Effective for Time Series Forecasting?" (DLinear as the linear control). Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997) for the small-sample correction.
