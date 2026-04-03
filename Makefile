.PHONY: install lint typecheck test test-unit test-integration train chat scale docker clean help

PYTHON := python
PIP := pip
SRC := src/nanochat
TESTS := tests

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PIP) install -e ".[dev]"

lint:  ## Run linter
	ruff check $(SRC) $(TESTS) scripts/
	ruff format --check $(SRC) $(TESTS) scripts/

format:  ## Format code
	ruff format $(SRC) $(TESTS) scripts/

typecheck:  ## Run type checker
	mypy $(SRC)

test: ## Run all tests
	pytest $(TESTS) -v --cov=$(SRC) --cov-report=term-missing

test-unit:  ## Run unit tests only
	pytest $(TESTS)/unit/ -v

test-integration:  ## Run integration tests
	pytest $(TESTS)/integration/ -v -s

test-scaling:  ## Run scaling tests
	pytest $(TESTS)/scaling/ -v

train:  ## Train nano model (smoke test)
	$(PYTHON) scripts/train.py --model-size nano --total-steps 100 --dtype float32

train-small:  ## Train small model
	$(PYTHON) scripts/train.py --model-size small --total-steps 5000

chat:  ## Launch interactive chat
	$(PYTHON) scripts/chat.py --model-size nano

evaluate:  ## Run evaluation
	$(PYTHON) scripts/evaluate.py --model-size nano --benchmark

scale:  ## Run scaling experiments
	$(PYTHON) scripts/run_scaling.py --experiment scale_n --max-steps 50

docker:  ## Build Docker image
	docker build -f docker/Dockerfile -t nanochat-jax:latest .

docker-gpu:  ## Build GPU Docker image
	docker build -f docker/Dockerfile.gpu -t nanochat-jax:gpu .

clean:  ## Clean build artifacts
	rm -rf outputs/ checkpoints/ .mypy_cache/ .ruff_cache/ dist/ *.egg-info/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
