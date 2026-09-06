.PHONY: help setup data features train backtest report test clean

PY ?= python
CONFIG ?= config/base.yaml

help:
	@echo "make setup     install pinned dependencies"
	@echo "make data      download the archive, build and validate the panel"
	@echo "make test      run the leakage, alignment and causality suite"
	@echo "make train     baselines + ARIMA + ridge + LightGBM on next-day direction"
	@echo "make vol       baselines + GARCH on next-day volatility"
	@echo "make deep      LSTM, CNN and DLinear on next-day direction"
	@echo "make backtest  cost sweep and equity curves for the newest run"
	@echo "make report    summary tables and figures for the newest run"
	@echo "make all       data, test, train, vol, deep, backtest, report"
	@echo "make clean     remove generated results and caches, keep raw data"

setup:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m src.data.build_panel --config $(CONFIG)

test:
	$(PY) -m pytest tests/ -q

train:
	$(PY) -m src.train --config $(CONFIG) \
		--model config/models/ridge.yaml \
		--model config/models/lightgbm.yaml \
		--model config/models/arima.yaml

vol:
	$(PY) -m src.train --config $(CONFIG) --target vol_1d \
		--model config/models/garch.yaml

deep:
	$(PY) -m src.train --config $(CONFIG) \
		--model config/models/lstm.yaml \
		--model config/models/cnn.yaml \
		--model config/models/dlinear.yaml

rolling:
	$(PY) -m src.train --config $(CONFIG) --scheme rolling \
		--model config/models/lightgbm.yaml

backtest:
	$(PY) -m src.run_backtest --config $(CONFIG)

report:
	$(PY) -m src.evaluate.report

all: data test train vol deep backtest report

clean:
	rm -rf reports/results reports/figures reports/data_quality.md
	rm -rf data/interim data/processed
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
