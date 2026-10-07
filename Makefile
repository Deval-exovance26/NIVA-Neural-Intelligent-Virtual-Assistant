# =============================================================================
# NIVA — top-level Makefile
# =============================================================================
# Entry points for the EC2 g6e.4xlarge (L40S, 48GB, CUDA 12.x) in us-east-2.
# Goal: fresh `git clone` -> `make setup && make verify-env` with zero friction.
#
#   make setup         create the offline-pipeline conda env (python 3.10)
#   make verify-env    assert GPU visible + torch.cuda.is_available() + print driver/CUDA
#   make fetch-models  download FLAME/GAGAvatar/ARTalk weights (scripts/fetch_models.sh)
#   make stage-assets  push NIVA capture assets to S3 (scripts/stage_assets_to_s3.sh)
#   make test          run offline-pipeline scaffold unit tests (no GPU needed)
#   make docker-build  build the CUDA 12.x offline-pipeline image
#
# Note: `fetch-models` and `stage-assets` invoke scripts owned by the other
# scaffold stage (scripts/). They are called by path and may not exist yet.
# =============================================================================

# Use bash with strict flags for every recipe.
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

CONDA_ENV   := niva-offline
OFFLINE_DIR := offline-pipeline
SCRIPTS_DIR := scripts

# conda run wrapper so recipes work without an interactive `conda activate`.
CONDA_RUN := conda run -n $(CONDA_ENV)

.DEFAULT_GOAL := help
.PHONY: help setup verify-env fetch-models stage-assets test docker-build

help: ## Show this help.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Create the conda env from offline-pipeline/environment.yml and install the package.
	@echo ">> Creating conda env '$(CONDA_ENV)' (python 3.10, CUDA 12.1 torch for L40S)..."
	conda env create -f $(OFFLINE_DIR)/environment.yml || \
		conda env update -f $(OFFLINE_DIR)/environment.yml --prune
	@echo ">> Installing niva-offline package (editable)..."
	$(CONDA_RUN) pip install -e $(OFFLINE_DIR)
	@echo ">> Done. Next: make verify-env"

verify-env: ## Assert GPU visible + torch CUDA available; print driver/CUDA/GPU.
	@echo ">> Verifying GPU + CUDA torch stack on this host..."
	$(CONDA_RUN) python -m niva_offline.verify_env

fetch-models: ## Download model weights (FLAME/GAGAvatar/ARTalk) via scripts/fetch_models.sh.
	@if [ -f "$(SCRIPTS_DIR)/fetch_models.sh" ]; then \
		echo ">> Running $(SCRIPTS_DIR)/fetch_models.sh ..."; \
		bash "$(SCRIPTS_DIR)/fetch_models.sh"; \
	else \
		echo "!! $(SCRIPTS_DIR)/fetch_models.sh not found."; \
		echo "   (Provided by the data/scripts scaffold stage.)"; \
		exit 1; \
	fi

stage-assets: ## Upload NIVA capture assets to S3 via scripts/stage_assets_to_s3.sh.
	@if [ -f "$(SCRIPTS_DIR)/stage_assets_to_s3.sh" ]; then \
		echo ">> Running $(SCRIPTS_DIR)/stage_assets_to_s3.sh ..."; \
		bash "$(SCRIPTS_DIR)/stage_assets_to_s3.sh"; \
	else \
		echo "!! $(SCRIPTS_DIR)/stage_assets_to_s3.sh not found."; \
		echo "   (Provided by the data/scripts scaffold stage.)"; \
		exit 1; \
	fi

test: ## Run offline-pipeline scaffold unit tests (CPU-only, no weights).
	$(CONDA_RUN) pytest $(OFFLINE_DIR)/tests -v

docker-build: ## Build the offline-pipeline CUDA 12.x image (run on EC2 with docker + nvidia toolkit).
	docker build -t niva-offline:latest $(OFFLINE_DIR)
