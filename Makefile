.PHONY: help setup data test lint train vol deep rolling backtest report all clean

PY ?= python
CONFIG ?= config/base.yaml
# The run that backs the direction and backtest tables in the README. Pinned
# because the default is the newest directory by modification time, which
# reorders whenever a report or a backtest is written. Override on the command
# line to point report and backtest at a different run.
RUN ?= reports/results/6246329f52

help:
	@echo "make setup     install pinned dependencies (plus ruff for make lint)"
	@echo "make data      download the archive, build and validate the panel"
	@echo "make test      run the leakage, alignment and causality suite"
	@echo "make lint      ruff over src/ and tests/"
	@echo "make train     baselines + ARIMA + ridge + LightGBM, pooled and per asset, on next-day direction"
	@echo "make vol       baselines + GARCH on next-day volatility"
	@echo "make deep      LSTM, CNN and DLinear on next-day direction"
	@echo "make backtest  cost sweep and equity curves for $(RUN)"
	@echo "make report    summary tables and figures for $(RUN)"
	@echo "make all       data, test, train, vol, deep, backtest, report"
	@echo "make clean     remove caches and the derived panel; committed results are kept"

setup:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m src.data.build_panel --config $(CONFIG)

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m ruff check src tests

train:
	$(PY) -m src.train --config $(CONFIG) \
		--model config/models/ridge.yaml \
		--model config/models/ridge_per_asset.yaml \
		--model config/models/lightgbm.yaml \
		--model config/models/lightgbm_per_asset.yaml \
		--model config/models/arima.yaml

vol:
	$(PY) -m src.train --config $(CONFIG) --target vol_1d \
		--model config/models/garch.yaml

deep:
	$(PY) -m src.train --config $(CONFIG) \
		--model config/models/lstm.yaml \
		--model config/models/lstm_per_asset.yaml \
		--model config/models/cnn.yaml \
		--model config/models/cnn_per_asset.yaml \
		--model config/models/dlinear.yaml \
		--model config/models/dlinear_per_asset.yaml

rolling:
	$(PY) -m src.train --config $(CONFIG) --scheme rolling \
		--model config/models/lightgbm.yaml

backtest:
	$(PY) -m src.run_backtest --config $(CONFIG) --run $(RUN)

report:
	$(PY) -m src.evaluate.report --run $(RUN)

all: data test train vol deep backtest report

# reports/results and reports/figures are committed and cited by the README,
# so clean leaves them alone. Delete a run directory by hand if you mean to.
clean:
	rm -rf reports/data_quality.md
	rm -rf data/interim data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
