.PHONY: install lint test train backtest serve docker clean

install:        ## Install the package with dev dependencies
	pip install -e ".[dev]"

lint:           ## Static checks
	ruff check src tests

test:           ## Run the test suite with coverage
	pytest --cov=power_forecaster --cov-report=term-missing

train:          ## Train + backtest + log to MLflow
	ppf train --model lightgbm

backtest:       ## Compare the model against the seasonal-naive baseline
	ppf backtest --model lightgbm

serve:          ## Run the prediction API locally
	ppf serve --reload

docker:         ## Build and run the full stack (API + MLflow)
	docker compose up --build

clean:
	rm -rf artifacts mlruns .pytest_cache .ruff_cache **/__pycache__
