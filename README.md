# Forecasting daily cryptocurrency returns, per asset

Seven model families against four naive baselines on daily Binance bars for BTC, ETH and LTC, evaluated walk-forward over 27 quarterly folds (2020-01 to 2026-07, 2,403 scored days per asset, 7,209 in total) with purging, embargo, per-fold scaling, and a Diebold-Mariano test on every comparison. Every learned family is fitted two ways, once on all three assets pooled and once per asset, and every result is reported per asset as well as pooled.

**Direction: nothing beat abstaining, on any asset, in either scope.** The best pooled model (ridge) called next-day direction 51.43% of the time against a 51.42% base rate. LightGBM's edge over its own base rate is +0.2pp on BTC, -0.2pp on ETH, +0.6pp on LTC, all inside a 3.6-point fold-to-fold spread. Fitting one model per asset did not help. For ridge and LightGBM the per-asset arm has lower accuracy than the pooled arm on all three assets; for the LSTM, CNN and DLinear it has higher accuracy, by 0.3 to 0.8 points, which is inside one seed's worth of noise, and no better Brier. Either way it trades a three-fold sample for parameters that had nothing asset-specific to learn, and neither scope beats abstaining on any asset.

**Volatility: GARCH beats the trailing mean on every asset.** EGARCH(1,1) has lower QLIKE than a 63-day trailing mean on BTC, ETH and LTC, each at p < 0.002. An earlier version of this README said the opposite. Splitting the result by asset showed why: GARCH(1,1) had never produced a BTC forecast in any fold. Its Student-t fit is integrated on BTC, the stationarity guard rejected it, and the rejected asset fell back to a hard-coded constant that scored like a calibration failure. That bug and a mismatch between the test's loss and the headline metric are both fixed below.

**Backtest: buy and hold wins.** Traded long-short as an equal-weight book, the LightGBM signal earned 10.0%/yr at zero cost and -19.3% at realistic taker fees; holding the three coins returned 34.5%. Per asset, the signal is positive after costs on LTC alone (+13.2% against 1.0% for holding LTC) and loses on BTC and ETH. One asset in three, on 27 folds, with no correction for the number of things compared, is the shape a chance result takes.

---

## How the models were fitted

This is the question the previous README did not answer. The data is a long panel, one row per asset-date, and a model can be fitted on all of it at once or on one asset's rows at a time.

| family | pooled arm | per-asset arm |
|---|---|---|
| ridge / L2 logistic, LightGBM, LSTM, 1D CNN, DLinear | one model per fold on BTC+ETH+LTC rows; the asset enters as cross-asset features (and, for the sequence models, an embedding) | one model per asset per fold, wrapped by `src/models/per_asset.py`, named `<family>_per_asset` |
| ARIMA, GARCH, EGARCH | n/a | always per asset; a return series has no pooled form |
| baselines (zero, persistence, historical mean, majority class) | n/a | per row, so per asset by construction |

The per-asset configs (`config/models/*_per_asset.yaml`) inherit every hyper-parameter from the pooled file through `model.extends` and change only `fit_scope`, so a gap between the two arms is a difference in scope and nothing else. An asset with fewer than 500 training rows in a fold would be skipped rather than fitted on scraps; with SOL gone, no asset ever is, so both arms are scored on identical rows. The per-fold scaler is still fitted on all training rows together; it sees no test row and is monotone, so it does not carry information across assets, but a stricter design would fit it per asset too.

**Why SOL is gone.** Solana lists on Binance in August 2020 and has no feature-complete rows until November. A per-asset model has nothing to train on for the first three folds and does not reach 500 rows until 2022. Rather than carry two asset universes through every table, the study is the three assets with full histories.

## Direction: next day, 27 folds, 2,403 predictions per asset

Fold-mean accuracy, pooled across assets. The DM column is a Diebold-Mariano test on Brier scores against the abstaining forecast, with the Harvey-Leybourne-Newbold small-sample correction.

| model | accuracy | base rate | ROC-AUC | MCC | folds beating base rate | DM p vs abstain |
|---|---|---|---|---|---|---|
| ridge | 0.5143 (SD 0.0430) | 0.5142 | 0.5221 | 0.0272 | 13 of 27 | 0.023 (worse) |
| zero (abstain at 0.5) | 0.5142 | 0.5142 | 0.5000 | 0.0000 | n/a | n/a |
| lightgbm | 0.5140 (SD 0.0362) | 0.5142 | 0.5239 | 0.0263 | 15 of 27 | 0.66 |
| lightgbm_per_asset | 0.5110 (SD 0.0348) | 0.5142 | 0.5167 | 0.0164 | 12 of 27 | 0.36 |
| historical mean | 0.5089 | 0.5142 | 0.4985 | 0.0100 | 7 of 27 | <0.001 |
| ridge_per_asset | 0.5070 (SD 0.0392) | 0.5142 | 0.5183 | 0.0106 | 13 of 27 | 0.22 |
| majority class | 0.4901 | 0.5142 | 0.4722 | -0.0409 | 1 of 27 | <0.001 |
| arima | 0.4806 | 0.5142 | 0.4720 | -0.0455 | 4 of 27 | <0.001 |
| persistence | 0.4724 | 0.5142 | 0.4616 | -0.0586 | 6 of 27 | <0.001 |

No model is significantly better than a constant 0.5. Ridge is significantly worse: it has the second-best ROC-AUC and still loses on Brier, because AUC asks whether up-days are ranked above down-days and Brier asks whether the printed number is a probability. Ridge's per-fold Brier ranges 0.240 to 0.288 against the abstainer's 0.250 by construction. It is confident at the wrong times.

**Persistence at 47.2% is a real effect, not noise.** Daily crypto returns mean-revert slightly at a one-day horizon, so betting that tomorrow repeats today loses more often than a coin flip.

### Per asset

Rows pooled across folds and seeds. Each asset has its own base rate; the DM test is run inside that asset's 2,403 rows.

| model | asset | accuracy | base rate | edge | ROC-AUC | Brier | DM p vs abstain |
|---|---|---|---|---|---|---|---|
| lightgbm | BTC | 0.5103 | 0.5081 | +0.22pp | 0.5255 | 0.2497 | 0.59 |
| lightgbm | ETH | 0.5149 | 0.5173 | -0.24pp | 0.5190 | 0.2506 | 0.65 (worse) |
| lightgbm | LTC | 0.5183 | 0.5127 | +0.56pp | 0.5287 | 0.2495 | 0.48 |
| ridge | BTC | 0.5148 | 0.5081 | +0.67pp | 0.5176 | 0.2521 | 0.084 (worse) |
| ridge | ETH | 0.5156 | 0.5173 | -0.17pp | 0.5208 | 0.2522 | 0.087 (worse) |
| ridge | LTC | 0.5173 | 0.5127 | +0.46pp | 0.5295 | 0.2506 | 0.63 (worse) |
| lightgbm_per_asset | BTC | 0.5029 | 0.5081 | -0.52pp | 0.5158 | 0.2505 | 0.71 (worse) |
| lightgbm_per_asset | ETH | 0.5102 | 0.5173 | -0.71pp | 0.5189 | 0.2509 | 0.51 (worse) |
| lightgbm_per_asset | LTC | 0.5153 | 0.5127 | +0.26pp | 0.5229 | 0.2508 | 0.60 (worse) |
| ridge_per_asset | BTC | 0.5031 | 0.5081 | -0.50pp | 0.5165 | 0.2502 | 0.86 (worse) |
| ridge_per_asset | ETH | 0.5085 | 0.5173 | -0.88pp | 0.5219 | 0.2508 | 0.50 (worse) |
| ridge_per_asset | LTC | 0.5094 | 0.5127 | -0.33pp | 0.5220 | 0.2514 | 0.27 (worse) |

Nothing is significant in the direction of skill on any asset. If less liquid markets were more predictable, LTC should lead and BTC should trail; LTC does lead by half a point, which is a sixth of the fold-to-fold spread. These twelve tests, plus the three per family below, are exploratory and uncorrected; the headline is the pooled test above. A Bonferroni threshold for the per-asset family would be p < 0.004, and nothing approaches it.

### Pooled fit against per-asset fit, same rows

Seeds averaged first, then compared on the rows both arms predicted. The DM statistic is per-asset against pooled: negative would mean the per-asset fit has the lower Brier.

| family | asset | accuracy pooled | accuracy per asset | DM stat | p | lower Brier |
|---|---|---|---|---|---|---|
| lightgbm | BTC | 0.5181 | 0.4994 | +1.07 | 0.29 | pooled |
| lightgbm | ETH | 0.5189 | 0.5139 | +0.35 | 0.73 | pooled |
| lightgbm | LTC | 0.5189 | 0.5135 | +1.39 | 0.16 | pooled |
| lightgbm | all | 0.5187 | 0.5089 | +1.68 | 0.093 | pooled |
| ridge | BTC | 0.5148 | 0.5031 | -2.57 | 0.010 | per asset |
| ridge | ETH | 0.5156 | 0.5085 | -2.34 | 0.019 | per asset |
| ridge | LTC | 0.5173 | 0.5094 | +0.95 | 0.34 | pooled |
| ridge | all | 0.5159 | 0.5070 | -2.01 | 0.044 | per asset |

The per-asset arm has lower accuracy everywhere and, for ridge, a significantly *better* Brier. Both are true: a ridge fitted on a third of the rows is regularised harder, prints probabilities closer to 0.5, and is punished less for being wrong. It is not more informative, it is less confident. The accuracies in this table differ slightly from the per-asset table above because a seed-averaged LightGBM forecast is a different object from the mean of five seed forecasts.

### Sequence models, 5 seeds each, same folds and rows

Run `2b7caa5555`: an LSTM, a 1D CNN and DLinear, each reading a 30-day window of the same features, each fitted pooled and per asset, five seeds per fold, about five hours of wall-clock time on CPU. Accuracy is the fold mean of the seed-averaged forecast with its fold-to-fold SD; seed SD is the within-fold spread across the five seeds, averaged over folds; ROC-AUC, MCC and Brier are over the pooled rows.

| model | accuracy | seed SD | base rate | ROC-AUC | MCC | Brier | folds beating base rate | DM p vs abstain |
|---|---|---|---|---|---|---|---|---|
| zero (abstain at 0.5) | 0.5142 | n/a | 0.5142 | 0.5000 | 0.0000 | 0.2500 | n/a | n/a |
| dlinear_per_asset | 0.5118 (SD 0.0247) | 0.0198 | 0.5142 | 0.5128 | 0.0191 | 0.2927 | 12 of 27 | <0.001 (worse) |
| lstm_per_asset | 0.5107 (SD 0.0239) | 0.0245 | 0.5142 | 0.5138 | 0.0170 | 0.2511 | 12 of 27 | 0.53 (worse) |
| lstm | 0.5056 (SD 0.0269) | 0.0237 | 0.5142 | 0.5087 | 0.0070 | 0.2511 | 12 of 27 | 0.20 (worse) |
| dlinear | 0.5038 (SD 0.0335) | 0.0220 | 0.5142 | 0.5016 | 0.0028 | 0.2766 | 12 of 27 | <0.001 (worse) |
| cnn_per_asset | 0.5007 (SD 0.0268) | 0.0189 | 0.5142 | 0.4978 | -0.0032 | 0.2521 | 11 of 27 | 0.002 (worse) |
| cnn | 0.4975 (SD 0.0285) | 0.0178 | 0.5142 | 0.4943 | -0.0087 | 0.2513 | 11 of 27 | 0.008 (worse) |

Nothing beats abstaining here either. Every arm has a significantly *worse* Brier than the constant 0.5 except the two LSTMs, which are indistinguishable from it (p = 0.20 and 0.53), and no arm beats the base rate on more than 12 of 27 folds. DLinear is the clearest case in the study of accuracy and calibration disagreeing: its per-asset arm has the highest fold-mean accuracy of the six sequence arms, 51.18%, and the worst Brier in the study, 0.293, because a linear map with a sigmoid on top prints probabilities near 0 and 1 (its forecasts span 0.002 to 0.999; the pooled LSTM's span 0.25 to 0.69). It is right about as often as a coin and certain every time. The seed SD is about two points, the same size as the gap between any two arms, so the order of the six rows is not stable across seeds; only DLinear's Brier is.

Per asset:

| model | asset | accuracy | base rate | edge | ROC-AUC | Brier | DM p vs abstain |
|---|---|---|---|---|---|---|---|
| lstm | BTC | 0.5057 | 0.5081 | -0.24pp | 0.5068 | 0.2513 | 0.28 (worse) |
| lstm | ETH | 0.5049 | 0.5173 | -1.24pp | 0.5053 | 0.2515 | 0.26 (worse) |
| lstm | LTC | 0.5094 | 0.5127 | -0.32pp | 0.5162 | 0.2506 | 0.97 (worse) |
| lstm_per_asset | BTC | 0.5088 | 0.5081 | +0.07pp | 0.5099 | 0.2510 | 0.63 (worse) |
| lstm_per_asset | ETH | 0.5074 | 0.5173 | -0.99pp | 0.5076 | 0.2519 | 0.20 (worse) |
| lstm_per_asset | LTC | 0.5169 | 0.5127 | +0.42pp | 0.5252 | 0.2503 | 0.54 |
| cnn | BTC | 0.5076 | 0.5081 | -0.05pp | 0.4937 | 0.2512 | 0.20 (worse) |
| cnn | ETH | 0.4940 | 0.5173 | -2.33pp | 0.4944 | 0.2513 | 0.15 (worse) |
| cnn | LTC | 0.4971 | 0.5127 | -1.56pp | 0.4941 | 0.2516 | 0.056 (worse) |
| cnn_per_asset | BTC | 0.5088 | 0.5081 | +0.07pp | 0.5120 | 0.2507 | 0.73 (worse) |
| cnn_per_asset | ETH | 0.5008 | 0.5173 | -1.65pp | 0.4815 | 0.2535 | 0.006 (worse) |
| cnn_per_asset | LTC | 0.4972 | 0.5127 | -1.55pp | 0.4977 | 0.2522 | 0.044 (worse) |
| dlinear | BTC | 0.4962 | 0.5081 | -1.19pp | 0.4996 | 0.2749 | <0.001 (worse) |
| dlinear | ETH | 0.5024 | 0.5173 | -1.49pp | 0.4987 | 0.2780 | <0.001 (worse) |
| dlinear | LTC | 0.5097 | 0.5127 | -0.30pp | 0.5071 | 0.2767 | <0.001 (worse) |
| dlinear_per_asset | BTC | 0.5152 | 0.5081 | +0.71pp | 0.5180 | 0.2853 | <0.001 (worse) |
| dlinear_per_asset | ETH | 0.5024 | 0.5173 | -1.49pp | 0.4998 | 0.2975 | <0.001 (worse) |
| dlinear_per_asset | LTC | 0.5150 | 0.5127 | +0.23pp | 0.5190 | 0.2955 | <0.001 (worse) |

The one cell in the direction of skill is the per-asset LSTM on LTC, +0.42pp with a DM statistic of -0.61, p = 0.54: nothing.

Pooled fit against per-asset fit, seeds averaged first, same rows, DM statistic per-asset against pooled as in the table above:

| family | asset | accuracy pooled | accuracy per asset | DM stat | p | lower Brier |
|---|---|---|---|---|---|---|
| lstm | BTC | 0.5056 | 0.5123 | -0.99 | 0.32 | per asset |
| lstm | ETH | 0.5056 | 0.5102 | +0.39 | 0.70 | pooled |
| lstm | LTC | 0.5164 | 0.5102 | -0.94 | 0.35 | per asset |
| lstm | all | 0.5092 | 0.5109 | -0.83 | 0.41 | per asset |
| cnn | BTC | 0.5135 | 0.5031 | -1.28 | 0.20 | per asset |
| cnn | ETH | 0.4965 | 0.4998 | +2.38 | 0.018 | pooled |
| cnn | LTC | 0.5035 | 0.5065 | +0.60 | 0.55 | pooled |
| cnn | all | 0.5045 | 0.5031 | +1.39 | 0.16 | pooled |
| dlinear | BTC | 0.5040 | 0.5148 | +2.55 | 0.011 | pooled |
| dlinear | ETH | 0.5060 | 0.5089 | +3.36 | <0.001 | pooled |
| dlinear | LTC | 0.5135 | 0.5214 | +4.23 | <0.001 | pooled |
| dlinear | all | 0.5078 | 0.5151 | +5.90 | <0.001 | pooled |

For the sequence models the per-asset arm has the higher fold-mean accuracy for all three families, by 0.3 to 0.8 points, and the higher seed-averaged accuracy on seven of the nine family-asset cells, the opposite of ridge and LightGBM, and inside one seed SD either way. On Brier it reverses again: DLinear's per-asset arm is significantly worse, because it is even more saturated; the CNN's is worse and the LSTM's better, neither significantly. Neither scope finds anything the other missed.

### What the trees looked at

Feature importance is not stable enough to interpret. `px_ret_1` ranks 1st on one fold and 270th on another. By mean gain the top features were `px_ret_1`, `rng_close_position`, `xa_market_ret_lag1`, `cal_dow_cos` and `aux_fng`; six of the top twelve are cross-asset, the one place the design expected to find something, and two are calendar sines, which is what a model finds when there is nothing else.

## Volatility: next-day log realised variance, 27 folds

QLIKE is the headline metric, and the Diebold-Mariano test runs under QLIKE on variances against the trailing mean. Lower QLIKE is better.

| model | QLIKE | MZ slope | MZ R2 | RMSE (log variance) | DM p vs trailing mean |
|---|---|---|---|---|---|
| EGARCH(1,1) | 0.772 | 1.361 | 0.074 | 1.075 | <0.001 (better) |
| GARCH(1,1), MZ-corrected | 0.790 | 0.569 | 0.040 | 1.326 | <0.001 (better) |
| EGARCH(1,1), MZ-corrected | 0.791 | 1.028 | 0.075 | 1.058 | <0.001 (better) |
| GARCH(1,1) | 0.792 | 0.789 | 0.037 | 1.337 | <0.001 (better) |
| trailing mean, 63d | 1.015 | 0.914 | 0.034 | 1.096 | n/a |
| persistence | 1.157 | 0.265 | 0.097 | 1.190 | 0.021 (worse) |

| model | asset | QLIKE | MZ slope | MZ R2 | DM p vs trailing mean |
|---|---|---|---|---|---|
| EGARCH(1,1) | BTC | 0.833 | 1.299 | 0.037 | 0.001 (better) |
| EGARCH(1,1) | ETH | 0.702 | 2.021 | 0.111 | <0.001 (better) |
| EGARCH(1,1) | LTC | 0.817 | 2.859 | 0.078 | <0.001 (better) |
| GARCH(1,1) | BTC | 0.877 | 4.047 | 0.025 | 0.013 (better) |
| GARCH(1,1) | ETH | 0.705 | 1.187 | 0.083 | <0.001 (better) |
| GARCH(1,1) | LTC | 0.800 | 2.231 | 0.054 | <0.001 (better) |
| trailing mean, 63d | BTC | 1.115 | 0.944 | 0.024 | n/a |
| trailing mean, 63d | ETH | 0.925 | 1.146 | 0.039 | n/a |
| trailing mean, 63d | LTC | 1.056 | 1.396 | 0.027 | n/a |

**The GARCH(1,1) BTC row is not a GARCH forecast.** With Student-t innovations the BTC fit comes out integrated (omega 0, alpha + beta 1.000) in all 27 folds, and the stationarity guard in `src/models/garch.py` rejects it. What is scored is the fallback: the training-window variance, a constant per fold. That constant still beats a 63-day trailing mean on QLIKE, because QLIKE punishes under-prediction far harder than over-prediction and a short trailing mean under-predicts before every spike. EGARCH, whose stability condition is on beta alone, fits BTC in 23 of 27 folds. Whether an integrated GARCH should be used for one-step forecasts anyway, as RiskMetrics does, is a decision this project has not taken; see "Not done".

**QLIKE and log squared error disagree, and that is not a contradiction.** GARCH has the lower QLIKE and the higher log RMSE on every asset. A forecast that is too high by a factor k costs log k + 1/k - 1 under QLIKE and (log k)^2 under squared log error; one that is too low by the same factor costs k - log k - 1 and the same (log k)^2. QLIKE cares which way you were wrong. The two MZ-corrected variants, calibrated on training folds only, land within 0.02 of the raw fits on QLIKE, so this is not an artifact a better calibration would remove.

## Backtest: long-short, 2% dead zone

The rule is per asset, on the predicted probability of an up move: long at `p > 0.52`, short at `p < 0.48`, flat in between. Positions are unit sized, then equal weighted across whichever assets hold one that day, so each date's weights sum to one in absolute terms. Turnover is charged per asset as the change in position, starting from a flat book. Volatility targeting exists in the engine but is off for these results. Binance spot taker fees are around 10 bps one way, so the 20 bps row is the realistic one.

| signal | cost (bps round trip) | annual return | Sharpe | max drawdown | annual turnover |
|---|---|---|---|---|---|
| lightgbm | 0 | 9.99% | 0.46 | -77.2% | 310 |
| lightgbm | 5 | 1.80% | 0.32 | -80.0% | 310 |
| lightgbm | 20 | -19.31% | -0.08 | -91.4% | 310 |
| lightgbm_per_asset | 0 | -1.89% | 0.28 | -90.2% | 354 |
| lightgbm_per_asset | 20 | -31.16% | -0.29 | -95.7% | 354 |
| ridge | 0 | 5.59% | 0.43 | -81.8% | 400 |
| ridge | 20 | -29.26% | -0.17 | -93.1% | 400 |
| ridge_per_asset | 0 | 0.67% | 0.34 | -89.6% | 368 |
| ridge_per_asset | 20 | -30.33% | -0.24 | -96.0% | 368 |
| buy and hold, equal weight | n/a | 34.49% | 0.78 | -77.6% | 0.15 |

### Per asset, LightGBM, each asset traded as its own book

These are standalone P&Ls, not each asset's share of the portfolio above.

| asset | cost (bps) | annual return | Sharpe | max drawdown | buy and hold |
|---|---|---|---|---|---|
| BTC | 0 | 8.80% | 0.41 | -63.0% | 38.98% |
| BTC | 20 | -10.52% | -0.01 | -79.6% | 38.98% |
| ETH | 0 | -21.67% | -0.06 | -87.7% | 49.70% |
| ETH | 20 | -36.12% | -0.38 | -96.2% | 49.70% |
| LTC | 0 | 37.60% | 0.82 | -74.4% | 0.96% |
| LTC | 20 | 13.23% | 0.53 | -77.8% | 0.96% |

LTC is the one cell in the study where a signal beats holding after costs. It is also the asset with the smallest accuracy edge that is still positive (+0.56pp), the one whose base rate the model most nearly matched, and one of three assets tested. The accuracy tables above say the LTC signal is not distinguishable from abstaining (p = 0.48). A backtest that wins where the forecast test does not is telling you about the size of a few moves, not about the model, and the right response is to note it and not to trade it.

## Data

Daily klines from the Binance public archive (`data.binance.vision`): a static host, no key, no rate limit, SHA256 checksum on every monthly zip. `src/data/fetch_binance.py` verifies all of them. No price data is committed; `make data` rebuilds it.

| asset | symbol | first bar | bars | first bar scored | bars scored |
|---|---|---|---|---|---|
| Bitcoin | BTCUSDT | 2017-08-17 | 3,271 | 2020-01-01 | 2,403 |
| Ethereum | ETHUSDT | 2017-08-17 | 3,271 | 2020-01-01 | 2,403 |
| Litecoin | LTCUSDT | 2017-12-13 | 3,153 | 2020-01-01 | 2,403 |

9,695 rows, no gaps, no duplicate timestamps, no OHLC ordering violations, no zero-volume bars. A bar is scored only inside a test window (first opens 2020-01-01) and at least 90 bars after its own listing, since `px_cumret_90` is the longest feature window.

**Cross-check.** BTC closes against Coinbase: 95.2% of 3,271 overlapping days agree within 1%. Every disagreement is in 2017–2020 (49 of 137 days in 2017, 71 in 2018, 35 in 2019, 2 in 2020, 0 of 2,038 from 2021 on). Binance quotes USDT and Coinbase quotes USD; the venues converged once arbitrage in the pair became routine. The disputed region sits almost entirely inside training warmup.

Two daily log returns exceed 50% in absolute value: ETH -0.59 and BTC -0.50 on 2020-03-12 (COVID crash). Real events, kept.

Two things to know if you write your own fetcher: the Binance archive switched from millisecond to microsecond timestamps mid-history, so the unit has to be detected per row, not per file; and prices are never forward-filled here, because a filled price is a zero return that drags every volatility estimate down. `reports/data_quality.md` regenerates all of this on every build.

## Features

80 features in 7 toggleable families: price (25: log returns at 6 lags, rolling moments, drawdown position), cross-asset (13: lagged BTC return, rolling correlation to BTC, cross-sectional rank), range (9: Parkinson, Garman-Klass, ATR, close position in bar), volume (9: quote volume z-score, taker buy ratio, trade count z-score), auxiliary (9: Fear and Greed, on-chain z-scores), technical (8: RSI, MACD, Bollinger position, stochastic, OBV z-score), calendar (7: sine/cosine day and month). The technical indicators are hand-rolled and add almost nothing: they are deterministic transforms of the same lagged prices the price family already holds.

## Design choices that separate a result from an artifact

**Returns, not prices.** Predicting tomorrow's price scores R² ≈ 0.99 by outputting today's. Every target in `src/data/targets.py` is a return, a direction, or a log variance.

**Walk-forward, never shuffled.** `src/splits/walk_forward.py`: train from 2018-01, test one quarter, step one quarter, 27 folds. Training always ends before testing begins.

**Scaler fitted inside the fold.** `src/preprocess.py` builds a fresh scaler per fold and never sees a test row. Standardising over the whole dataset leaks future volatility backward, does not crash, and just makes the numbers better than they should be.

**Purging and embargo are different things.** Purging removes training rows whose *label* overlaps the test period: one day, at a one-day horizon. Embargo drops a further 5 days so no training row sits directly against the boundary. Required gap 6 days; measured gap 7. `tests/test_no_leakage.py` asserts this at several embargo settings, including zero.

**The test's loss is the headline's loss.** Direction is tested on Brier, which is what the accuracy table is read against. Volatility is tested on QLIKE, which is what the volatility table ranks on. When those two were allowed to differ the tables contradicted each other about the same forecasts, and the contradiction looked like a finding.

**Both scopes, same rows.** A pooled fit and a per-asset fit are compared only on rows both predicted, and the number of those rows is printed next to the p-value.

## Bugs found while building this

Most were caught by measurement rather than by reading code, which is the argument for building the evaluation harness first, and two of them were caught only when the measurement was split by asset, which is the argument for never reporting a pooled number alone.

- **GARCH falling back to a hard-coded constant.** The training-variance fallback was recorded only for fits that passed the stationarity guard. A rejected fit, which is every BTC fold, fell to `log(1e-4)`, about a quarter of BTC's realised variance. Pooled, that read as QLIKE 7.9 and a story about calibration drift. Per asset, it read as BTC 8.9 against ETH 0.7, which is not a calibration problem. `tests/test_no_leakage.py` now forces the guard to reject and requires the fallback to equal the training variance.
- **Diebold-Mariano under the wrong loss for volatility.** Squared error on log variance said GARCH was significantly worse than the trailing mean while QLIKE in the next column said it was better. Both were computed correctly; the test now runs under the metric the table ranks on.
- **Per-asset tables that were never written.** `per_asset_results` and `per_asset_breakdown` were defined, documented, and called from nowhere. The four per-asset numbers in the earlier README were computed by hand and had no file behind them. Every run now writes `per_asset.csv` and `dm_per_asset.csv`, and the backtest writes `backtest_per_asset.csv`.
- **GARCH forecasting a constant.** `arch`'s `forecast()` returns a value only at the final observation unless passed `start`; every other test row silently fell back to a per-asset constant.
- **Diebold-Mariano comparing labels to probabilities.** Hard 0/1 labels against a constant 0.5 under squared loss made abstaining unbeatable by construction.
- **A volatility baseline in the wrong units.** The historical-mean baseline predicted a mean return of ~0.001 against a log-variance target of ~-7.
- **Rolling windows spanning two assets.** A missing `groupby(level='asset')`. Caught by comparing a single-asset build against the same asset's rows in a multi-asset build.
- **GARCH reading its own future through `arch`.** `fix()` derives its backcast, demeaning constant and variance bounds from the whole series handed to it, including the test window. `src/models/garch.py` now computes those from the training prefix and drives the recursion itself; the leakage suite perturbs the tail of a test window and requires every earlier forecast to come back bit-identical.

## Limitations

- **27 folds is not many.** Fold-to-fold SD of accuracy is ~3.6 points. The honest reading of the direction null is that no *large* edge exists, not that none does.
- **Three assets.** Heavily correlated, chosen for long clean histories, i.e. because they survived. Any asset-to-asset comparison is on a cross-section of three.
- **Per-asset tests are uncorrected.** Twelve per-asset direction tests, six per-asset volatility tests and eight scope comparisons are reported at face value and labelled exploratory. The pooled tests are the headline. For direction the omission is conservative because nothing survives; for volatility it is not needed because everything survives at p < 0.02, well inside any correction.
- **GARCH(1,1) on BTC is a constant.** See above. The per-asset volatility rows for BTC compare EGARCH, which fitted, against a fallback, which did not.
- **One horizon.** Everything is one day ahead.
- **USDT, not USD.** Deepest books and longest history, but the peg has broken briefly on a handful of days.
- **One hand-argued exception in the leakage scan.** `src/models/garch.py` is the only file under `src/` allowed a backward shift, because reading the variance recursion one row ahead is a genuine one-step forecast. The justification is written next to the allowlist entry in `tests/test_no_leakage.py`.

## Not done, named rather than omitted

Relaxing the GARCH stationarity guard to admit an integrated fit for one-step forecasting, which would give BTC a real GARCH(1,1) row. A per-asset scaler for the per-asset arm. Hourly bars (same archive, 24× the data). Point-in-time universe including delisted pairs. Horizon-decay curve. Order-book imbalance from the depth snapshots in the same archive. Triple-barrier labels, conformal intervals, regime-switching models. Optuna and MLflow: hyperparameters are frozen in YAML and runs are tracked by config hash in a CSV.

## Reproduce

```bash
pip install -r requirements.txt
make data      # 320 monthly files, checksum-verified, ~15 min first run, cached after
make test      # 104 tests, mostly leakage, alignment and per-asset contracts
make train     # baselines + ARIMA + ridge + LightGBM, pooled and per asset, on direction
make vol       # baselines + GARCH/EGARCH on volatility
make deep      # LSTM, 1D CNN, DLinear, pooled and per asset, 5 seeds each (about 5 h, CPU)
make backtest  # cost sweep, equity curves, per-asset books
make report    # summary tables, per-asset tables, figures
```

If `make` is not on your PATH, each target is one `python -m src.<module>` line in the `Makefile`.

Python 3.11+; developed on 3.14.3 with pandas 3.0.5, numpy 2.5.2, torch 2.14.0, all pinned. No GPU anywhere.

Every experiment is a YAML file, and the SHA256 of the merged config names its output directory under `reports/results/`. Those directories are committed, predictions included, so any number above can be checked without retraining.

| table | run | files |
|---|---|---|
| direction, pooled and per asset; backtest | `6246329f52` | `results.csv`, `per_asset.csv`, `dm.csv`, `dm_per_asset.csv`, `backtest.csv`, `backtest_per_asset.csv`, `report.md` |
| direction, sequence models | `2b7caa5555` | `results.csv`, `per_asset.csv`, `dm.csv`, `dm_per_asset.csv`, `report.md` |
| volatility | `015429d904` | `results.csv`, `per_asset.csv`, `dm.csv`, `dm_per_asset.csv`, `report.md` |

Every model in a table was evaluated on the same folds and rows as the baselines beside it. `--dm-baseline` changes the comparison without changing the config hash, so reporting a different baseline does not relocate the results.

```
config/            base.yaml plus one file per model family, and a *_per_asset.yaml for each
src/
  data/            fetchers, validation suite, panel builder, targets
  features/        one module per family, plus the registry
  splits/          purged and embargoed walk-forward splitter
  models/          baselines, ARIMA, GARCH, ridge, LightGBM, LSTM, CNN, DLinear, per-asset wrapper
  evaluate/        metrics, Diebold-Mariano, reporting
  backtest/        cost model and engine, written from scratch
  train.py         experiment runner
tests/             alignment, leakage, causality and per-asset checks
notebooks/         exploration and results analysis, not the pipeline
```

## References

López de Prado, *Advances in Financial Machine Learning*, ch. 7 (purging and embargo). Zeng et al. (2022), "Are Transformers Effective for Time Series Forecasting?" (DLinear as the linear control). Diebold and Mariano (1995); Harvey, Leybourne and Newbold (1997) for the small-sample correction. Patton (2011), "Volatility forecast comparison using imperfect volatility proxies" (QLIKE).
