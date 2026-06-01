---
name: run-experiment
description: Launch a med-vision-transformers training run (src/modeling/train.py) with a chosen task / dataset / loss / CV config, then capture and report the MLflow run id, the cross-validation metrics, and the held-out test metrics. Use when the user wants to train a model, kick off an MIM pretrain → classification finetune, or reproduce an experiment.
---

# Run an experiment (med-vision-transformers)

Build the right `python -m src.modeling.train` command, confirm it, run it, and report the
MLflow run id + metrics so the run is reproducible and can be evaluated / referenced from
`.env` (`RUN_ID_*`).

## Arguments

`$ARGUMENTS` — free-form: a task (`mim` | `classification` | `multi-task`), a dataset
(`coco` | `ultrasound` | `brats` | `brats_slice`), a loss, and any override
(`--num-splits`, `--num-epochs`, `--lr`, `--apply-mmd`, `--filter-ood`,
`--pre-trained-model-run-id <id>`, ...). Anything unspecified uses the `train.main` defaults.

## Flow

### Step 1: Resolve the configuration

Read `src/modeling/train.py:main` for the current option names/defaults and `src/config.py`
for `N_SPLITS`, `NUM_EPOCHS`, `LR`, etc. Confirm the data the run needs exists:

```bash
python -m src.modeling.train --help
ls data/ models/ 2>/dev/null
```

If the required dataset directory is missing (COCO under
`data/external/opg111.v1i.coco-segmentation/`, BRATS, ultrasound), say so and stop — the run
will fail without data. For a finetune (`--pre-trained-model-run-id`), confirm
`models/<id>/base_model.pth` exists.

### Step 2: Show the command and confirm

Show the exact command, e.g.:

```bash
python -m src.modeling.train \
  --training-task multi-task --data-type coco \
  --regression-loss-type MSE --classification-loss-type BCEWithLogits \
  --num-splits 5 --num-epochs 20
```

Note that training is **CPU by default** unless `DEVICE=cuda`/`mps` is exported (see
`utils.DEVICE`). For a real run this can be long — confirm with the user before launching
(offer `/verify` first for a quick smoke check). Ask whether to run in the background.

### Step 3: Run it

Launch the command (foreground, or background for a long run). Stream/inspect the loguru
output. The run id is logged near the start:

```
MLflow run ID: <run_id>
```

Capture it. Training writes checkpoints to `models/<run_id>/` (`fold_*_model.pth`,
`base_model.pth`, `model.pth`).

### Step 4: Report

Report:
- the **MLflow run id** and `models/<run_id>/` artifacts;
- the cross-validation summary (`avg_*` + the `*_lower`/`*_upper` t-CI) and, for
  classification/multi-task, the `test_*` held-out metrics (PR-AUC is the headline);
- to inspect: `mlflow ui --backend-store-uri mlruns` (or `python -m src.modeling.eval.eval_run --parent-run-id <id> --metric pr_auc`);
- to evaluate on the held-out test set:
  `python -m src.modeling.eval.eval_clf --parent-run-id <id> --training-task <task>`;
- if this run is a reusable artifact (e.g. an MIM pretrain to finetune from, or a model for
  `predict.py`), the matching `RUN_ID_*` line to add to `.env` (see `.env.example` /
  `utils.get_model_run_id`).

Do not commit anything (`models/` and `mlruns/` are gitignored and hook-blocked).
