# Forecasting daily cryptocurrency returns

A walk-forward evaluation of seven model families against four naive baselines, on daily bars for BTC, ETH, LTC and SOL, from August 2017 to July 2026. The question is whether any of them beat those baselines once you refuse to let them see the future and once you charge them for trading.

The short answer is no, and the interesting part is the shape of the failure.

## What it found

**Nothing beat the baseline on direction.** Across 27 quarterly walk-forward folds, the best model (LightGBM) predicted next-day direction correctly 51.35% of the time against a base rate of 51.12%. That is an edge of 0.23 percentage points against a fold-to-fold standard deviation of 3.6 points. A Diebold-Mariano test on Brier scores gives a p-value of 0.97 against the abstaining baseline, and it is one of only two models the test cannot separate from abstaining, the LSTM being the other at p = 0.20. Everything else, including the ridge regression that posted the best ROC-AUC of anything here, is significantly worse than a constant 0.5.

**The seed matters more than the architecture.** Averaged over 27 folds, the LSTM's edge over the abstaining baseline is -0.53 percentage points. Its standard deviation across five random seeds, within a fold, is 2.61 points. The noise from reseeding is five times the size of the effect being measured, which means a single-seed run of any of these models could report almost anything.

**The edge that did exist died at 5 basis points.** Traded as a long-short strategy, LightGBM's signal returned 8.6% annually at zero cost, 0.2% at 5 bps round trip, and -21.4% at 20 bps. It turned the portfolio over 323 times a year. Buy-and-hold on the same four assets over the same window returned 56.2% at a Sharpe of 0.98. The model never came close. The ridge signal did worse still, losing 1.85% a year before any costs at all, which is the more honest ending: the model that ranked best was not merely expensive to trade, it was unprofitable to trade.

**GARCH lost to a 63-day moving average.** This one surprised me. Volatility is supposed to be the tractable target, and GARCH(1,1) is supposed to be the tool. On QLIKE loss, a trailing mean of log realised variance scored 1.009 and GARCH scored 7.93. The reason is not that GARCH lacks information: per asset, its forecasts correlate 0.46 to 0.53 with next-day realised variance, about the same as persistence. The problem is that its calibration drifts. Its per-fold bias has a standard deviation of 1.34 in log variance, against 0.02 for persistence, so the level is wrong in a different direction on every fold and the ranking never gets a chance to pay off.

**Feature importance is not stable enough to interpret.** `xa_ref_ret_lag1` ranks 2nd on one fold and 53rd on another. Any story told about which features matter here would be a story about which fold you happened to look at.

## Results

### Next-day direction, 27 folds, 9,299 predictions per model

| model | accuracy | base rate | ROC-AUC | MCC | folds beating base rate | DM p vs abstain |
|---|---|---|---|---|---|---|
| lightgbm | 0.5135 (SD 0.0357) | 0.5112 | 0.5298 | 0.0414 | 16 of 27 | 0.97 |
| ridge / L2 logistic | 0.5115 (SD 0.0418) | 0.5112 | 0.5315 | 0.0397 | 14 of 27 | 0.001 |
| zero (abstain at 0.5) | 0.5112 | 0.5112 | 0.5000 | 0.0000 | n/a | n/a |
| historical mean | 0.5044 | 0.5112 | 0.4548 | -0.0425 | 7 of 27 | <0.001 |
| majority class | 0.4875 | 0.5112 | 0.5000 | 0.0000 | 0 of 27 | <0.001 |
| arima | 0.4833 | 0.5112 | 0.4936 | -0.0080 | 4 of 27 | <0.001 |
| persistence | 0.4771 | 0.5112 | 0.4630 | -0.0539 | 7 of 27 | <0.001 |

The last column is a Diebold-Mariano test on Brier scores against the abstaining forecast, with the Harvey-Leybourne-Newbold small-sample correction. Every model in this table except LightGBM is significantly *worse* than a constant 0.5. LightGBM is the only one here the test cannot separate from abstaining, and it does not beat it either. The sequence models are tested the same way in their own table below.

An L2 logistic regression, with its penalty chosen on the inner validation slice, scored the highest ROC-AUC of anything in the study: 0.5315 against LightGBM's 0.5298. Whatever weak structure is present in these 80 features is linear, and the boosted trees are not finding anything on top of it.

That AUC does not survive contact with the Brier score, and the gap between the two is the most useful thing in this table. ROC-AUC only asks whether the model ranks up-days above down-days. Brier asks whether the number it prints is a probability. Ridge ranks better than anything else here and still loses to abstaining at p = 0.001, because it is confident at the wrong times: its per-fold Brier ranges from 0.240 to 0.285 while the abstaining forecast sits at exactly 0.250 by construction. A model that ranks well and is calibrated badly is worth less than one that says nothing, and reporting only the AUC would have hidden that completely.

Persistence scoring 47.7% is not noise. Daily crypto returns mean-revert slightly at a one-day horizon, so betting that tomorrow repeats today loses more often than a coin flip.

The three sequence models were run separately with five seeds each, on the same folds and the same rows:

| model | accuracy | seed SD within fold | ROC-AUC | MCC | folds beating base rate |
|---|---|---|---|---|---|
| lstm | 0.5060 (SD 0.0352) | 0.0261 | 0.5205 | 0.0225 | 12 of 27 |
| dlinear | 0.5048 (SD 0.0329) | 0.0228 | 0.5134 | 0.0166 | 13 of 27 |
| cnn | 0.4968 (SD 0.0269) | 0.0181 | 0.5044 | -0.0012 | 10 of 27 |

None of them reached the 51.12% base rate. Diebold-Mariano against the abstaining baseline puts DLinear and the CNN significantly worse (p below 0.001) and the LSTM indistinguishable (p = 0.20). DLinear is a single linear layer over a decomposed input and it finished within 0.12 percentage points of the LSTM while training in a fifth of the time, 3.6 seconds per fit against 7.6 for the LSTM and 13.5 for the CNN. That is the Zeng et al. point reproduced on this dataset: the recurrent and convolutional machinery bought nothing here. Taken with the ridge result, the ordering across all seven families is close to the reverse of their complexity.

Per asset, LightGBM's accuracy minus that asset's own base rate: BTC +0.70pp, LTC +0.66pp, SOL +0.40pp, ETH -0.43pp. The gradient does not match the efficiency story. If less liquid markets were more predictable, SOL should lead and BTC should trail, and neither happens.

By regime, LightGBM's edge over the base rate is +0.42pp in bear markets, +0.46pp sideways, and +0.04pp in bull markets. All three sit well inside the fold-to-fold spread.

### Next-day volatility, 27 folds

| model | QLIKE | MZ slope | MZ R2 | RMSE (log variance) |
|---|---|---|---|---|
| trailing mean, 63d | 1.009 | 1.034 | 0.056 | 1.086 |
| persistence | 1.097 | 0.323 | 0.128 | 1.164 |
| EGARCH(1,1) | 5.709 | 0.555 | 0.082 | 1.388 |
| GARCH(1,1) | 7.929 | 0.558 | 0.065 | 1.688 |

The Mincer-Zarnowitz columns show the split cleanly. Persistence has the highest R2, meaning its forecasts track the variation best, but a slope of 0.32 means they are badly scaled. The trailing mean has a slope of 1.03, meaning it is almost perfectly calibrated, and it wins on QLIKE by being unbiased rather than by being informative.

Two corrections were tried for GARCH, both fitted on training folds only: a constant level shift and a full intercept-plus-slope regression. They produced results within 1% of each other, so the gap is not a calibration artifact that a better correction would remove.

### Backtest, long-short with a 2% dead zone

| signal | cost (bps round trip) | annual return | Sharpe | max drawdown | annual turnover |
|---|---|---|---|---|---|
| lightgbm | 0 | 8.59% | 0.44 | -77.9% | 323 |
| lightgbm | 5 | 0.16% | 0.31 | -81.4% | 323 |
| lightgbm | 20 | -21.42% | -0.09 | -92.3% | 323 |
| ridge | 0 | -1.85% | 0.34 | -85.5% | 416 |
| ridge | 5 | -11.55% | 0.19 | -88.6% | 416 |
| ridge | 20 | -35.30% | -0.25 | -95.7% | 416 |
| buy and hold | n/a | 56.16% | 0.98 | -83.0% | 0.23 |

Binance spot taker fees are around 10 bps one way, so the realistic row for each signal is the 20 bps one, not the 0 bps one.

Ridge is the sharper test of the two, and it fails earlier. LightGBM at least starts positive and is then eaten by costs, which is the ordinary way a weak signal dies. Ridge loses 1.85% a year *before* anyone charges it anything, while trading 29% more than LightGBM does. Its ranking ability is real, in the sense that the AUC is the best in the study, but the dead zone converts that ranking into positions at the wrong sizes and the wrong moments. The best-ranking model in the study is the worse of the two strategies at every cost level, which is a cleaner statement of the paper's point than the LightGBM curve alone: the problem is not that the edge is small, it is that ranking skill this weak does not survive being turned into a trade.

The Sharpe column disagrees with the return column on the ridge rows, and the reason is worth stating because it is easy to report the flattering half by accident. Sharpe is built from the arithmetic mean of daily returns; annual return is compounded. Ridge at zero cost has an arithmetic mean of +24.0% a year and an annualised volatility of 71.2%, so the variance drag of roughly half the squared volatility, 25.3%, is larger than the mean and the compounded result comes out negative. LightGBM survives the same arithmetic only because it is less volatile, 60.8%, and starts from a higher mean. The compounded number is the one an account would actually experience, so it is the one to read.

### Where these numbers came from

Each table above is one run directory under `reports/results/`, named by the hash of its config.

| table | run | command |
|---|---|---|
| direction, ridge, LightGBM and ARIMA | `a08f75962a` | `make train` |
| direction, sequence models | `a2e2e68684` | `src.train --model lstm.yaml --model cnn.yaml --model dlinear.yaml` |
| volatility | `96037aadcc` | `src.train --target vol_1d --model garch.yaml` |
| backtest | `a08f75962a` | `src.run_backtest --run reports/results/a08f75962a` |

Every model in a given table was evaluated on the same folds and the same rows as the baselines printed beside it. The direction table used to be stitched from two runs, one for ridge and one for LightGBM and ARIMA, which meant the two strongest models had never been compared on the same frame. `make train` now runs all three together, so a single directory holds every number in that table and the Diebold-Mariano column is a comparison rather than a juxtaposition. The earlier directories are still on disk and still reproduce their own rows.

`dm.csv` in each direction run holds the Diebold-Mariano results, so every p-value quoted above has a file behind it rather than a calculation someone did once and threw away. The training run writes it directly; the sequence-model directory predates that and was backfilled from its own stored predictions, which is the same function over the same rows. Pass `--dm-baseline` to compare against something other than the abstaining forecast. That flag deliberately does not enter the hashed config, because changing which comparison you report should not change where the results are written.

## Why the design looks the way it does

If you have not worked with financial time series, three of the choices below are the ones that usually get made wrong, and they are the difference between a result and an artifact.

**Returns, not prices.** Train a model to predict tomorrow's Bitcoin price and it will score an R-squared near 0.99. It has learned to output today's price, which is an excellent predictor of tomorrow's price and worth nothing. Predicting the return removes the level and leaves only the part nobody knows. Every target in `src/data/targets.py` is a return, a direction, or a log variance.

**Walk-forward, never shuffled.** A shuffled train/test split puts January 2024 in training and December 2023 in test, which lets the model learn from the future. Training here always ends before testing begins. The splitter is in `src/splits/walk_forward.py` and produces 27 folds: train from 2018-01, test one quarter, step forward one quarter.

**Fit the scaler inside the fold.** Standardising features across the whole dataset before splitting leaks future volatility backward, because the mean and standard deviation used to scale a 2019 row were computed partly from 2025 data. `src/preprocess.py` builds a fresh scaler per fold and never shows it a test row. This is the single most common bug in this category of project and it is invisible in the output: it does not crash, it just makes the numbers better than they should be.

There is a fourth choice worth spelling out because the usual explanation of it is wrong.

**Purging and embargo are different things.** Purging removes training rows whose *label* overlaps the test period. With a one-day horizon, the last training row's label is drawn from the first test day, so exactly one day gets purged. Long feature windows do not need purging, however long they are, because a 63-day rolling mean ending at `t` contains nothing from after `t`. Embargo is separate: it drops a further 5 days so that a training row is not sitting directly against the test boundary, where serial correlation makes it nearly the same observation. The required gap between the last training date and the first test date is therefore 6 days. The measured gap is 7, because the splitter keeps rows strictly before the boundary rather than up to it. `tests/test_no_leakage.py` asserts the requirement at several embargo settings, including zero, where the purge alone still removes the horizon.

## Getting it running

```bash
pip install -r requirements.txt
make data      # downloads 392 monthly files, verifies checksums, builds the panel
make test      # 84 tests, mostly leakage and alignment checks
make train     # baselines + ARIMA + ridge + LightGBM on direction
make vol       # baselines + GARCH/EGARCH on volatility
make deep      # LSTM, 1D CNN, DLinear, 5 seeds each
make backtest  # cost sweep and equity curves
make report    # summary tables and figures
```

`make data` takes about 15 minutes on a first run and is cached afterwards. Everything else runs on CPU. `make train` takes a few minutes. `make deep` took 56 minutes for 27 folds by 5 seeds by 3 architectures, almost all of it spent fitting. No GPU is used anywhere.

The notebooks need `jupyter` and `ipykernel`, which are not in `requirements.txt` because nothing in the pipeline imports them.

Python 3.11 or newer. Developed and run on 3.14.3 with pandas 3.0.5, numpy 2.5.2 and torch 2.14.0, all pinned in `requirements.txt`.

## The data

Daily klines from the Binance public archive at `data.binance.vision`, which is a static file host with no key and no rate limit. Every monthly zip ships a SHA256 checksum and `src/data/fetch_binance.py` verifies all of them.

| asset | symbol | first bar | bars |
|---|---|---|---|
| Bitcoin | BTCUSDT | 2017-08-17 | 3,271 |
| Ethereum | ETHUSDT | 2017-08-17 | 3,271 |
| Litecoin | LTCUSDT | 2017-12-13 | 3,153 |
| Solana | SOLUSDT | 2020-08-11 | 2,181 |

11,876 rows total, no gaps, no duplicate timestamps, no OHLC ordering violations, no zero-volume bars.

One thing in that fetcher is worth knowing about if you write your own. Binance switched the archive from millisecond to microsecond timestamps partway through, and a single symbol's history contains both. Detecting the unit once from the first row parses every later bar as a date in the year 56971. The unit has to be decided per row.

Prices are never forward filled. A filled price creates a zero return, which drags every volatility estimate down.

### Cross-checking against a second exchange

BTC closes were compared against Coinbase, and 95.2% of 3,271 overlapping days agree within 1%. That looks like a failure against the 99% target until you break it down by year:

| period | days outside 1% |
|---|---|
| 2017 | 49 of 137 |
| 2018 | 71 of 365 |
| 2019 | 35 of 365 |
| 2020 | 2 of 366 |
| 2021 to 2026 | 0 of 2,038 |

Every disagreement is in the first three years. Binance quotes USDT and Coinbase quotes USD, and the two venues only converged once cross-exchange arbitrage in the pair became routine. The first test fold opens in January 2020, so the disputed region sits almost entirely inside the training warmup.

Three daily log returns exceed 50% in absolute value. All three trace to real events: ETH at -0.59 and BTC at -0.50 on 2020-03-12, and SOL at -0.55 on 2022-11-09. Those are the COVID crash and the FTX collapse, not data errors.

Coin Metrics community tier serves BTC, ETH and LTC but not SOL, so the on-chain features are missing for one of the four assets. The fetcher narrows its request per asset rather than losing the whole source, and reports what it could not get.

`reports/data_quality.md` is regenerated on every build and holds all of this.

## Features

80 features in 7 toggleable families, so you can turn any group off and rerun.

| family | count | examples |
|---|---|---|
| price | 25 | log returns at 6 lags, rolling moments, drawdown position |
| cross asset | 13 | lagged BTC return, rolling correlation to BTC, cross-sectional rank |
| range | 9 | Parkinson and Garman-Klass estimators, ATR, close position in bar |
| volume | 9 | quote volume z-score, taker buy ratio, trade count z-score |
| auxiliary | 9 | Fear and Greed index, on-chain activity z-scores |
| technical | 8 | RSI, MACD, Bollinger position, stochastic, OBV z-score |
| calendar | 7 | day of week, day of month, month, as sine/cosine pairs |

The technical indicators are hand-rolled in `src/features/technical.py` rather than pulled from a library, and it is worth saying plainly that they add almost nothing. RSI, MACD and the rest are deterministic transforms of the same lagged price series that the price family already contains. They contribute nonlinearity, not information.

By mean gain, the top features were `px_ret_1`, `rng_close_position`, `xa_market_ret_lag1` and `xa_market_dispersion`. Four of the top twelve are cross-asset, which is the one place the design expected to find something. It was not enough.

## Repository layout

```
config/            base.yaml plus one file per model family
src/
  data/            fetchers, validation suite, panel builder, targets
  features/        one module per family, plus the registry that assembles them
  splits/          purged and embargoed walk-forward splitter
  models/          baselines, ARIMA, GARCH, ridge, LightGBM, LSTM, CNN, DLinear
  evaluate/        metrics, Diebold-Mariano, reporting
  backtest/        cost model and engine, written from scratch
  train.py         the experiment runner
tests/             alignment, leakage and causality checks
notebooks/         exploration and results analysis, not the pipeline
```

Every experiment is defined by a YAML file, and the SHA256 of the merged config names the output directory. `reports/results/runs.csv` indexes every run so a number from three weeks ago can still be traced to the settings that produced it.

`data/` is gitignored in full. No price data is committed anywhere in this repo, and `make data` rebuilds it from scratch.

## Bugs found while building this

Most of these were caught by measurement rather than by reading the code, which is the argument for building the evaluation harness before the models. The last one was not, which is the argument for reading it anyway.

**GARCH forecasting a constant.** `arch`'s `forecast()` returns a value only at the final observation unless you pass a `start` argument. Every other test row silently fell back to a per-asset constant. The symptom was a forecast that looked plausibly scaled while correlating -0.05 with realised variance.

**A degenerate GARCH fit on SOL.** Fitted on SOL's short and extremely volatile early history, the model returned parameters with alpha plus beta above 1, which has no finite unconditional variance. It scored a correlation of -0.34 while the other three assets sat near +0.5, and it dragged the pooled number down far enough to look like a general failure. There is now a stationarity guard in `src/models/garch.py`.

**Diebold-Mariano comparing labels to probabilities.** The first version scored hard 0/1 labels against a constant 0.5 forecast under squared loss. A label scores 0 or 1 and the constant always scores 0.25, so the abstaining baseline came out unbeatable by a wide margin. It was measuring the output encoding, not the forecasts.

**A volatility baseline that was a category error.** The historical mean baseline predicts a mean return of about 0.001 against a log variance target of about -7. It was not a weak forecast, it was the wrong units, and it made everything else look good by comparison.

**Rolling windows spanning two assets.** Caught by a subagent's own check: truncating one asset's history cannot detect a missing `groupby(level='asset')`, because the other asset's rows are untouched. Comparing a single-asset build against the same asset's rows in a multi-asset build does detect it.

**GARCH reading its own future through `arch`.** The one-step-ahead forecast comes from freezing the training parameters and running the variance recursion over the concatenated train and test returns. The recursion is causal, since the variance for `t+1` is a function of the squared innovation at `t`. But `arch`'s `fix()` is more than the recursion: it derives the backcast that seeds it, the constant it demeans by, and the variance bounds that clamp it from whatever series it is handed, which here included the test window. Multiplying the tail of a test window by twenty moved the variance at the start of that recursion by about two percent. The influence decays geometrically in beta and is gone by the test rows, so nothing in the tables above depends on it: re-running the volatility experiment after the fix left 96 percent of the predictions bit-identical and moved the rest, and every metric in the table, by no more than 2e-15. It was a dependence on future data all the same. `src/models/garch.py` now computes those three quantities from the training prefix and drives the recursion itself, and `tests/test_no_leakage.py` perturbs the tail of a test window and requires every earlier forecast to come back bit-identical.

## What this does not do

Named rather than quietly omitted:

- **Hourly bars.** The archive has them and the config would take them. 24 times the data would change the signal-to-noise ratio and possibly the conclusions.
- **Gramian Angular Field encodings for the 2D CNN.** The 1D dilated convolution is implemented; the image-encoding variant is not.
- **Optuna and MLflow.** Hyperparameters are set in YAML and frozen. Runs are tracked by config hash in a CSV. This is worse than a real tracking setup and better than nothing.
- **Triple-barrier labelling, conformal intervals, regime-switching models.** All are more aligned with what a trader actually needs than "did it go up".
- **Order book data.** Depth snapshots are in the same archive and order book imbalance is one of the few features with documented short-horizon power. It is a much larger data engineering job.

## Limitations

**Survivorship bias.** BTC, ETH, LTC and SOL were chosen because they have long clean histories, which means they were chosen because they survived. A model that fails on survivors would probably fail worse on the full universe, so the direction of this bias happens to be conservative here. It would not be if the result had been positive.

**Four assets is a small cross-section.** Crypto returns are heavily correlated, so four assets provide considerably less than four times the information of one. The cross-sectional features in particular are working with a very thin panel.

**One horizon.** Everything predicts one day ahead. Skill typically decays with horizon and the shape of that decay is a real result this project does not have.

**USDT, not USD.** Binance USDT pairs have the deepest books and the longest history, which is why they are the primary source, but USDT has broken its peg briefly on a handful of occasions and those days carry noise that a USD pair would not have.

**One hand-argued exception in the leakage scan.** `src/models/garch.py` is the only file under `src/` allowed to shift backwards in time, because reading the variance recursion one row ahead is a genuine one-step-ahead forecast rather than a peek. That argument is made by hand rather than enforced by the pattern, so it is written out in `tests/test_no_leakage.py` next to the allowlist entry and backed by two tests specific to that model. Any new negative shift anywhere in `src/` fails the suite until someone justifies it the same way.

**Twenty-seven folds is not many.** The fold-to-fold standard deviation of directional accuracy is around 3.6 percentage points. Detecting a real edge of half a point against that noise needs far more folds than this study has, so the honest reading of the null result is that no large edge exists, not that no edge exists.

## References

The methodology follows López de Prado, *Advances in Financial Machine Learning*, particularly chapter 7 on cross-validation with purging and embargo. The DLinear model is from Zeng et al. (2022), "Are Transformers Effective for Time Series Forecasting?", included specifically because a near-linear model is the right control for whether the LSTM and CNN earn their complexity. Diebold and Mariano (1995) for the predictive accuracy test, with the Harvey, Leybourne and Newbold small-sample correction.
