# med-vision-transformers

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Vision transformers for medical data research

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── pyproject.toml     <- Poetry project metadata, dependencies
│                         ([tool.poetry.dependencies] + dev group) and tool config
├── poetry.lock        <- Pinned, resolved dependency versions (committed)
├── poetry.toml        <- Local Poetry config (in-project .venv)
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── data_processing.py  <- Datasets, transforms, masking helpers
    │   ├── models.py           <- ViT heads (MIM / classification / multi-task)
    │   ├── utils.py            <- Device, transforms, losses, shared helpers
    │   ├── predict.py          <- Attention / MIM-reconstruction visualization
    │   ├── segmentation.py     <- DINOv2 + k-means segmentation
    │   ├── train.py            <- Train models (k-fold CV + held-out test)
    │   └── eval
    │       ├── eval_clf.py     <- Evaluate a trained run on the held-out test set
    │       └── eval_run.py     <- Confidence interval for a metric across folds
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Set up project

```bash
poetry install            # runtime + dev deps into an in-project .venv
poetry run pre-commit install
poetry run pre-commit run --all-files
```

Run commands inside the environment with `poetry run ...` (e.g. `poetry run python -m
src.modeling.train --help`) or `poetry shell`.

Copy `.env.example` to `.env` and populate the `RUN_ID_*` identifiers with the MLflow run
ids of your trained runs (used by `src/modeling/predict.py`). The `models/` and `mlruns/`
directories hold weights and tracking data and are not version-controlled.

## Experiment protocol

Training (`src/modeling/train.py`) uses **stratified k-fold cross-validation** on the
training pool and keeps a **held-out test set** untouched (the dataset's official COCO test
split, or a freshly carved stratified split for the other datasets) for a single final
evaluation. Cross-validation metrics are reported as a mean with a t-based 95% confidence
interval; classification is a single-logit binary task (PR-AUC is the headline metric for
the imbalanced caries detection). Inputs are ImageNet-normalized and MIM reconstructs in
that normalized space, computing its loss only on masked patches.
