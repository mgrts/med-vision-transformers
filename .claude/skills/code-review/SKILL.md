---
name: code-review
description: Review pending changes in the med-vision-transformers repo for correctness and the silent-bug classes this research codebase actually hits — MIM masking & masked-only reconstruction loss, input normalization & the no-sigmoid recon contract, the single-logit binary head, data leakage / held-out-test / stratified-(group-)k-fold protocol, best-epoch/best-fold selection, the MLflow-arg + seed + t-based-CI reproducibility contract, dataset label construction (BRATS censoring, ultrasound unmatched, brats_slice groups), model constructor signatures across train/predict/eval, dependency-import hygiene, and secrets/large-file/artifact hygiene. Read-only by default; surfaces findings grouped by severity. Use before every commit, or via /commit-push.
---

# Code review for med-vision-transformers

Review the changes currently in the working tree (staged + unstaged + untracked) against
the standards that matter for THIS codebase. There is **no test suite** and the model runs
on CPU by default, so most bugs here are **silent**: a masking arg that disables MIM, a lost
normalization, a leaked test set, or a reintroduced 2-class head changes the math /
experiment validity / metrics without raising. The invariants below are the canonical list
(kept in sync with `CLAUDE.md` → "CRITICAL invariants").

The review is **read-only by default** — fixes are surfaced as recommendations and only
applied if the user explicitly asks.

## Arguments

`$ARGUMENTS` — optional. Specific files or globs to scope the review (defaults to the entire
working-tree diff).

## Flow

### Step 1: Gather changes

```bash
git status --short
git diff --staged --stat
git diff --stat
```

If there is nothing pending, stop: "Nothing to review."

### Step 2: Read the diff

For each changed file, read the actual diff (not just the file list). Note which subsystems
are touched — that selects which checks below apply and which subagent to delegate to.

### Step 3: Delegate deep audits to subagents

When the diff touches a fragile subsystem, dispatch the matching subagent (Agent tool,
`subagent_type`) and fold its findings into the report. Run independent subagents in parallel.

- Touches masking / MIM / `MIMHead` / transforms / normalization / `MultiTaskLoss` /
  `compute_mmd` (`models.py`, `data_processing.py` mask code, MIM branches of `train.py`,
  transforms in `utils.py`) → **`mim-ssl-reviewer`**.
- Touches `build_datasets` / `holdout_split` / `kfold_splits` / `train_model` / `main` /
  `eval/*.py` / dataset labels & groups → **`data-split-leakage-auditor`**.
- Touches `models.py` heads / the model factory / head construction in
  `predict.py` / `eval_clf.py` / label dtype-shape / `pyproject.toml` deps / top-level imports
  → **`model-contract-reviewer`**.

For a small diff that clearly matches none, do the checks inline.

### Step 4: CRITICAL — MIM masking & reconstruction (see `mim-ssl-reviewer`)

- `create_mask(batch_size, image_size, patch_size, mask_ratio)` is called with the **model
  patch size** (`model.base_model.config.patch_size`, = 8), never image width (width → a
  silent no-op that masks 0 patches).
- `get_masked_images` returns `(masked_images, mask)`; `mask` is **True for KEPT** pixels;
  masked region filled with **0.0** (post-norm mean).
- MIM loss is **masked-only**: `criterion(mim_output[~mask], images[~mask])` (mim) and
  `MultiTaskLoss` masks internally. Not a full-image reconstruction.
- `MIMHead` has **no final sigmoid** (reconstructs in normalized space). `predict.py` fills
  with 0.0 and `denormalize`s for visualization.

### Step 5: CRITICAL — Input normalization contract

- `TRAIN_TRANSFORM` / `EVAL_TRANSFORM` / `TRAIN_TRANSFORM_SIMPLIFIED` all end with
  `Normalize(IMAGENET_MEAN, IMAGENET_STD)`; train & eval share FOV (0.875 ratio); no vertical
  flip; `segmentation.py` DINOv2 preprocessing also normalized. Removing normalization
  silently breaks both the pretrained-backbone input space and the MIM target/mask-fill.

### Step 6: CRITICAL — Single-logit binary head

- `num_classes = 1` everywhere; dataset `__getitem__` returns a **float `[1]`** label;
  `collate_fn` → `[B, 1]`; classifier returns `[B, 1]` logits; loss is
  `BCEWithLogitsLoss(pos_weight=...)`. No `[CARIES, background]` one-hot, no `long` labels,
  no macro averaging. Metrics via `compute_binary_metrics` on the positive class
  (PR-AUC headline + ROC-AUC + F1 at a tuned threshold).

### Step 7: CRITICAL — Data leakage / held-out test / CV (see `data-split-leakage-auditor`)

- No `train+val+test` concatenation + `random_split`. Official COCO test stays separate;
  non-COCO test carved by `holdout_split` and never in a CV fold.
- `kfold_splits` = `StratifiedKFold` (`StratifiedGroupKFold` when `groups` present, e.g.
  `brats_slice`); train folds use `pool_aug`, val/test use `pool_eval`.
- `train_model` restores best-epoch weights; `main` exports the best fold (lowest val loss);
  `eval_clf.py` evaluates the held-out test only.

### Step 8: CRITICAL — Reproducibility / MLflow / statistics

- `main` seeds `random`+`numpy`+`torch`(+cuda) from `RANDOM_STATE`; splits seeded.
- `mlflow.log_params` logs the **actual CLI args** (`num_epochs`/`lr`/`weight_decay`), not
  the config constants.
- CIs via `confidence_interval` (**t-based**, one shared helper) — never `mean ± 2*std`.
- Every `torch.load(..., map_location=DEVICE)`.

### Step 9: HIGH — Dataset labeling & model signatures

- BRATS drops censored (`ALIVE`/non-numeric) survival rows; ultrasound skips unmatched
  files; `BRATSSliceDataset` pairs scan/seg by patient dir + exposes `groups`.
- `MIMTransformer(base_model, dropout_rate)` — no `image_size` kwarg;
  `predict.load_mim_model(dino_model, path)` 2 args; `MultiTaskTransformer` built with
  `num_classes=1` in predict/eval. (See `model-contract-reviewer`.)

### Step 10: HIGH — Dependency & duplicate hygiene

- Every third-party import is declared in `pyproject.toml` `[tool.poetry.dependencies]` (dev
  tooling in the `dev` group); grep imports vs the file. There is no `requirements.txt`.
- No re-introduced duplicate helper (`compute_mmd` only in `utils.py`; masking only in
  `data_processing.py`; `confidence_interval`/`compute_binary_metrics`/`collect_predictions`
  single-definition).
- `apply_mmd` operates on CLS features (off by default, documented same-dataset OOD).

### Step 11: MEDIUM — Style, hygiene & docs drift

- black + isort at **99** (`pyproject.toml [tool.black]`, isort `--profile black`);
  `make format` then `make lint` leave no diff. (Note the known tooling line-length
  inconsistency — flake8 131 in pre-commit vs 99 elsewhere; flag, don't silently "fix".)
- Dependency/build management is **Poetry** (`[tool.poetry]` + `poetry.lock`, `poetry-core`
  backend); no PEP 621 `[project]` table, no `requirements.txt`.
- **No secrets / no PII** in the diff. Nothing staged under `data/`, `models/`, `mlruns/`,
  `reports/`, `notebooks/`, nor any `*.nii/*.npy/*.pt/*.pth/*.ckpt/*.pkl/*.db/*.ipynb`/
  `.env`; no `git add -f` past `.gitignore`; no file `> 10 MB`.
- New public function/class has a docstring + type hints; if an invariant changed, `CLAUDE.md`
  is updated.

### Step 12: Compile + lint

Run both and report exit status:

```bash
poetry run python -m compileall -q src   # no test suite; this catches syntax errors
make lint                                 # poetry run flake8 + isort --check + black --check (99)
```

A failing compile is a critical finding; lint failures are medium (`make format` fixes most).

### Step 13: Report findings

Group findings by severity:

- **Critical** — MIM masking no-op / loss-not-masked / re-added sigmoid / lost
  normalization; reintroduced 2-class head or wrong label dtype; test-set leakage / non-
  disjoint CV / eval-on-train; MLflow-arg or seed-contract break; invalid CI; undeclared
  dependency; broken compile.
- **High** — group-leakage (slices), best-epoch/best-fold regression, bad model constructor
  signature, dataset-labeling regression, duplicate-helper drift, `map_location` drop.
- **Medium** — style/hygiene/poetry-table, docs drift, lint failure.
- **Low** — comment / naming / docstring / type-hint polish.

For each finding: file path, the symbol or line, and a concrete suggestion. Do not make
changes unless the user asks.

If there are zero findings: report "Review passed — N files reviewed, M lines changed,
compile <result>, lint <result>."
