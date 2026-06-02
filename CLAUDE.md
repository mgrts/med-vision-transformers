# CLAUDE.md — med-vision-transformers

Research codebase for **vision transformers on medical imaging**. A DINO ViT-S/8 backbone
(`facebook/dino-vits8`, via HuggingFace `AutoModel`) is trained with three tasks —
**MIM** (masked image modeling, self-supervised), **binary classification**, and
**multi-task** (MIM + classification jointly) — across several datasets: dental panoramic
X-rays in COCO format (target **CARIES**), ultrasound (НОРМА/ПАТОЛОГИЯ), and BRATS MRI
(2-year survival + tumor-slice). Training runs as **stratified k-fold cross-validation with
a held-out test set**, logged to MLflow (`mlruns/`), with models saved under `models/<run_id>/`.

Most bugs in this repo are **silent**: there is no test suite, the code runs on CPU by
default, and a wrong masking arg / lost normalization / leaked test set / reintroduced
2-class head changes the math, the experiment validity, or the metrics without raising.
The `/code-review` skill and the three review subagents exist to catch exactly those.

## Package map

- `src/config.py` — single source of truth: paths, `RANDOM_STATE`, model/training
  hyperparameters, `IMAGENET_MEAN/STD`, `N_SPLITS`, `TEST_FRAC`, the BRATS labeling
  constants (`BRATS_SURVIVAL_THRESHOLD_DAYS/SURVIVAL_SLICE/SLICE_INDICES/TUMOR_AREA_THRESHOLD`),
  `TARGET_CATEGORIES = ['CARIES']`, OOD category lists, and the 12 `RUN_ID_*` env lookups
  (see `.env.example`).
- `src/dataset.py` — synthetic geometric-shapes generator (Typer). Not used by training.
- `src/plots.py` — COCO caries-mask visualization (Typer).
- `src/modeling/`
  - `models.py` — `MIMHead` (decoder, **no final sigmoid** — reconstructs in normalized
    space), `MultiLabelClassificationHead`, `MIMTransformer(base_model, dropout_rate)`,
    `MultiLabelClassificationTransformer(base_model, num_classes, dropout_rate)`,
    `MultiTaskTransformer(base_model, image_size, num_classes, dropout_rate)`.
  - `data_processing.py` — datasets (`ImageDatasetCOCO`, `ImageDataset` (ultrasound),
    `ImageDatasetBrats`, `BRATSSliceDataset`, `MultiLabelImageDataset`), `collate_fn`,
    `create_mask`, `get_masked_images` (returns `(masked, mask)`), `resolve_concrete`
    (unwraps `Subset`/`ConcatDataset`).
  - `utils.py` — `DEVICE` (autodetect + `DEVICE` env override), `TRAIN_TRANSFORM` /
    `EVAL_TRANSFORM` / `TRAIN_TRANSFORM_SIMPLIFIED` (+ `NORMALIZE`), `denormalize`,
    `confidence_interval` (t-based), regression + classification loss factories
    (`get_classification_loss_function(..., pos_weight=)`), `MultiTaskLoss` (masked-only
    MIM term), `compute_mmd` (RBF), `load_pretrained_model`, `get_model_run_id`.
  - `train.py` — the training entrypoint and most of the experiment logic:
    `build_datasets`, `holdout_split`, `kfold_splits`, `filter_ood_indices`,
    `compute_binary_metrics`, `select_threshold`, `collect_predictions`, `train_model`,
    `forward_batch`, `get_model`, `get_loss_function`, `build_mmd_pools`, `get_mmd_images`,
    `main`.
  - `predict.py` — attention-map + MIM-reconstruction visualization (Typer).
  - `segmentation.py` — DINOv2 + k-means segmentation utility.
  - `eval/eval_clf.py` — evaluate each fold of a training run on the **held-out test set**
    only; binary metrics + t-CI. (Imports `collect_predictions`/`compute_binary_metrics`
    from `train.py` — single source.)
  - `eval/eval_run.py` — t-CI for one MLflow metric across a parent run's nested folds.

## How to run

```bash
poetry install                # runtime + dev deps into an in-project .venv
poetry run pre-commit install
cp .env.example .env          # then fill RUN_ID_* for predict/eval

# Train (Typer; --help lists all options). Tasks: mim | classification | multi-task.
# Data: coco | ultrasound | brats | brats_slice. Prefix everything with `poetry run`.
poetry run python -m src.modeling.train --training-task multi-task --data-type coco --num-splits 5
poetry run python -m src.modeling.train --training-task mim --data-type coco            # SSL pretrain
poetry run python -m src.modeling.train --training-task classification \
    --pre-trained-model-run-id <mim_run_id>                                             # finetune

# Evaluate a finished run on the held-out test set:
poetry run python -m src.modeling.eval.eval_clf --parent-run-id <run_id> --training-task multi-task

# Visualize attention / MIM reconstruction:
poetry run python -m src.modeling.predict --model-type REAL_MULTITASK
```

Make targets (all Poetry-backed): `make requirements` (poetry install) · `make lint`
(flake8 + isort --check + black --check) · `make format` (isort + black) · `make data`
(synthetic generator) · `make clean`.
There is **no `make test`** — this repo has no test suite (the `/verify` skill smoke-runs
a 1-epoch/2-fold training instead).

## CRITICAL invariants

1. **MIM masking contract.** `create_mask(batch_size, image_size, patch_size, mask_ratio)`
   — the third arg is the **model patch size (8 for dino-vits8)**, NEVER the image width
   (passing width makes `224//224 = 1` patch and masks nothing — a silent no-op).
   `get_masked_images` (in `data_processing.py`) returns `(masked_images, mask)`, fills the
   masked region with **0.0** (the post-normalization mean), and `mask` is **True for KEPT
   (visible)** pixels. MIM loss is computed **only on masked pixels** —
   `criterion(mim_output[~mask], images[~mask])` for `mim`, and `MultiTaskLoss` masks
   internally. `predict.py` mirrors this (fill 0.0, then `denormalize` for saving).
2. **Normalized reconstruction.** `MIMHead` has **no final activation** (it reconstructs in
   the ImageNet-normalized input space). Do NOT re-add a sigmoid — it would clamp outputs to
   [0,1] while the targets are normalized (~[-2.6, 2.6]). The mask fill (0.0) is the
   post-norm mean precisely because of this.
3. **Input normalization.** All transforms (`TRAIN/EVAL/SIMPLIFIED`) end with
   `transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)` because the DINO backbone is
   ImageNet-pretrained. Train and eval share the same field-of-view (0.875 resize/crop
   ratio). No vertical flip (anatomically invalid for X-ray/MRI). `segmentation.py`'s
   DINOv2 preprocessing is also normalized.
4. **Single-logit binary head.** `num_classes = 1` everywhere. Every dataset's
   `__getitem__` returns a **float label tensor of shape `[1]`** (`0.` / `1.`); `collate_fn`
   stacks to `[B, 1]`. The classifier outputs `[B, 1]` logits; loss is
   `BCEWithLogitsLoss(pos_weight=...)`. There is **no 2-column `[CARIES, background]`
   one-hot** and **no macro averaging** — metrics are computed on the positive class via
   `compute_binary_metrics` (PR-AUC is the headline metric; ROC-AUC + F1 at a tuned
   threshold alongside). Reintroducing a background class or `dtype=long` labels breaks both
   shape and metrics.
5. **No data leakage / held-out test.** Never concatenate `train+val+test` and random-split
   it (the old critical bug). `build_datasets` keeps the official COCO **test split fully
   separate**; for the other datasets a stratified (group-aware) test set is carved by
   `holdout_split` and is **never** in any CV fold. Cross-validation uses `kfold_splits`
   (`StratifiedKFold`, or `StratifiedGroupKFold` when `groups` is present, e.g.
   `brats_slice`). Train folds index `pool_aug` (augmented); validation/test index
   `pool_eval` (deterministic). `eval_clf.py` evaluates on the held-out test **only**.
6. **Model selection & checkpointing.** `train_model` deep-copies the **best-epoch** weights
   (lowest val loss) and restores them before returning — the saved model matches the
   reported metrics. The exported model is the **best fold** (lowest val loss), NOT the fold
   "closest to the average".
7. **Reproducibility & stats.** `main` seeds `random` + `numpy` + `torch` (+cuda) from
   `RANDOM_STATE`; splits take a seeded `random_state`. MLflow logs the **actual CLI args**
   (`num_epochs`, `lr`, `weight_decay`), not the config-constant defaults. Confidence
   intervals use `confidence_interval` (**t-based**, `scipy.stats.t`) — never `mean ± 2*std`.
   This single helper lives in `utils.py` and is imported by `train.py` and both eval
   scripts. Every `torch.load` passes `map_location=DEVICE`.
8. **Dataset labeling.** BRATS survival **drops censored rows** (`ALIVE`/non-numeric
   `Survival_days`) instead of mislabeling them; threshold/slice are config constants.
   Ultrasound **skips** files matching no label substring (never silently positive).
   `BRATSSliceDataset` pairs each scan with its segmentation by **patient directory** (not
   positional `zip`) and exposes `groups` for patient-level CV. COCO labels are caries
   presence (no background class).
9. **Model constructor signatures.** `MIMTransformer(base_model, dropout_rate=...)` —
   accepts **no `image_size`** (it reads `base_model.config.image_size`).
   `MultiTaskTransformer(base_model, image_size, num_classes, dropout_rate)`. `predict.py`
   and `eval_clf.py` build heads with `num_classes=1`; `predict.load_mim_model(dino_model,
   path)` takes exactly 2 args.
10. **Domain adaptation (experimental, off by default).** `apply_mmd` computes MMD on the
    encoder **CLS features** of an in-domain vs an OOD batch (so it backprops). The OOD
    batch is drawn from the SAME dataset — a documented limitation, unvalidated, exploratory.
    `filter_ood` errors if it would remove every training sample. Don't claim MMD/filtering
    improves robustness without an OOD evaluation.
11. **Dependency / import hygiene.** Every third-party import must be declared in
    `pyproject.toml` under `[project.dependencies]` (PEP 621 runtime: `torch`, `torchvision`,
    `transformers`, `timm`, `mlflow`, `nibabel`, `pillow`, `numpy`, `pandas`, `pyarrow`,
    `scikit-learn`, `scipy`, `matplotlib`, `opencv-python`, `pycocotools`, `loguru`, `tqdm`,
    `typer`, `python-dotenv`) — dev tooling (`black`, `flake8`, `isort`, `pre-commit`,
    `mkdocs`) lives in the `[tool.poetry.group.dev.dependencies]` group. There is **no
    `requirements.txt`** — this is a Poetry project; install with `poetry install` and add
    deps with `poetry add` (resolved versions are pinned in `poetry.lock`). `scipy` backs the
    t-interval (also a `scikit-learn` dep). No dead duplicate helpers — `compute_mmd` lives
    only in `utils.py`,
    `get_masked_images`/`create_mask`/`resolve_concrete` only in `data_processing.py`,
    `confidence_interval`/`compute_binary_metrics`/`collect_predictions` have a single
    definition each.

## Dev workflow

- Formatting: **black + isort at line length 99** (`pyproject.toml [tool.black]`; isort
  `--profile black`). `make format` then `make lint` should leave no diff. The
  `auto_format` hook applies this to every edited `.py`.
- **Tooling line-length is inconsistent**: `pyproject [tool.black]` and `setup.cfg [flake8]`
  use **99**, but `.pre-commit-config.yaml`'s flake8 uses **131**. `setup.cfg` flake8 also
  ignores `E501`, so line length is effectively unenforced by lint. Keep black at 99; if you
  unify, align all three. (Flagging, not auto-fixing.)
- Dependency & build management is **Poetry** (`poetry-core` >= 2.0 backend). Static metadata
  and runtime deps live in the **PEP 621 `[project]`** table (`[project.dependencies]`); only
  Poetry-specific config stays under `[tool.poetry]` (`packages = [{include = "src"}]` and the
  `[tool.poetry.group.dev.dependencies]` dev group). In-project `.venv` via `poetry.toml`;
  resolved pins in `poetry.lock`. Bump `[project].version` for a release (`poetry version
  patch`). Install with `poetry install`, run via `poetry run`. (Re-run `poetry lock` after
  editing dependencies so the lock content-hash stays consistent.)
- Datasets are not in the repo; `models/`, `mlruns/`, `data/`, `reports/`, `notebooks/` are
  gitignored. Existing checkpoints are 2-class / sigmoid-era and will NOT load into the
  current single-logit / no-sigmoid models — retrain.

## What NOT to commit

`.gitignore` excludes `data/`, `models/`, `mlruns/`, `reports/`, `notebooks/`, `.env`. The
`block_large_secret` hook additionally refuses to stage anything in those dirs or any
`.nii/.npy/.pt/.pth/.ckpt/.pkl/.db/.ipynb/.env/.pem/.key` (and files > 10 MB), mirroring
pre-commit's `check-added-large-files` (maxkb=10000) + `detect-private-key`. **Never
`git add -f`** these. `.gitkeep` placeholders are allowed.

## Claude Code setup in this repo

- **Skills** (`.claude/skills/`): `/code-review` (read-only review of the working tree
  against the invariants above), `/commit-push` (gated commit → push to `main`),
  `/run-experiment` (launch a training run, capture the MLflow run id), `/verify` (smoke
  1-epoch/2-fold run), `/new-dataset` (scaffold + wire a new `Dataset`).
- **Subagents** (`.claude/agents/`): `mim-ssl-reviewer`, `data-split-leakage-auditor`,
  `model-contract-reviewer` — `/code-review` delegates to these for deep, file-specific audits.
- **Hooks** (`.claude/settings.json` → `.claude/hooks/`): auto-format edited `.py`
  (black+isort @ 99); guard against destructive git (force-push, `reset --hard`,
  `--no-verify`, deleting `main`, **any Claude/AI commit attribution**); block staging
  large/secret/artifact/medical-image files; byte-compile `src/` on stop. Disable any hook
  by editing `.claude/settings.json`.
- **Commit policy:** Claude is **never** listed as a commit author (no `Co-Authored-By`,
  no "Generated with Claude") — enforced by the `guard_git` hook.
