#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = med-vision-transformers
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python Dependencies (runtime + dev) into the Poetry-managed venv
.PHONY: requirements
requirements:
	poetry install




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8, isort, and black (use `make format` to do formatting)
.PHONY: lint
lint:
	poetry run flake8 src
	poetry run isort --check --diff --profile black src
	poetry run black --check --config pyproject.toml src

## Format source code with isort + black
.PHONY: format
format:
	poetry run isort --profile black src
	poetry run black --config pyproject.toml src




## Set up the Poetry-managed (in-project .venv) environment
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry env ready. Install deps with 'make requirements'; run with 'poetry run ...'."




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	poetry run python src/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
