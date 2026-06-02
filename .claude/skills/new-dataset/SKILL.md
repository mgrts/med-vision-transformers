---
name: new-dataset
description: Scaffold a new PyTorch Dataset in src/modeling/data_processing.py following this repo's conventions (single binary float-[1] labels, optional patient-level groups, transform passed in) and wire it into src/modeling/train.build_datasets + src/config.py + CLAUDE.md. Use when adding a new imaging dataset/modality for the classification / MIM / multi-task pipeline.
---

# Add a new dataset (med-vision-transformers)

Add a new imaging dataset so it plugs into the existing training pipeline without breaking
the binary-head, leakage-split, or labeling invariants. Match the existing dataset classes
(`ImageDatasetCOCO`, `ImageDataset`, `ImageDatasetBrats`, `BRATSSliceDataset`).

## Step 1: Gather requirements

Ask the user (only what you can't infer):
- **Name / modality** and `data_type` key (e.g. `xray2`, `ct`).
- **On-disk layout** and how to enumerate samples (file glob, an annotation/CSV, COCO).
- **How the binary label is derived** (positive = ?), and whether some items are
  **unlabeled / censored** (must be skipped/dropped, never silently coerced — cf. ultrasound
  unmatched-skip and BRATS censored-drop).
- **Grouping**: is there a patient/scan id so multiple items share a subject? If yes, the
  dataset must expose `self.groups` (one id per item) for patient-level CV.
- Whether the official source provides a **held-out test split** (like COCO) or the test set
  should be carved from a single pool (like the others).

## Step 2: Write the Dataset class (conventions — match existing classes)

In `src/modeling/data_processing.py`:

- `__init__(self, ..., transform=None)` stores `transform`, builds `self.image_paths` (or
  equivalent) and `self.labels`, and sets `self.num_classes = 1`.
- Labels: `self.labels` is a list of `[float]` (`[0.]` / `[1.]`). Skip/drop unlabeled or
  censored items (log a `logger.warning` with the count) — do **not** default them to a class.
- `__getitem__` returns `(image, label)` where `image = self.transform(PIL_RGB_image)` and
  `label = torch.tensor(self.labels[idx], dtype=torch.float32)` (shape `[1]`). Grayscale
  medical images: `Image.open(p).convert('RGB')`; volumes: min-max normalize the slice with a
  zero-division guard, then `Image.fromarray((s*255).astype(np.uint8)).convert('RGB')`.
- If grouped: build `self.groups` (a per-item subject id, e.g. an int index per scan), as
  `BRATSSliceDataset` does. Pair related files (scan/mask) by **directory**, not positional
  `zip`.
- Reuse `MASK_RATIO` / config constants; promote any new threshold/slice/path into
  `src/config.py` (no magic numbers in the dataset).

## Step 3: Wire into training

In `src/config.py`: add the data directory / annotation paths and any labeling constants.

In `src/modeling/train.py` `build_datasets(data_type)`: add a branch returning
`(pool_aug, pool_eval, y, groups, official_test)`:

- `pool_aug` = dataset with `TRAIN_TRANSFORM`; `pool_eval` = the SAME data with
  `EVAL_TRANSFORM` (identical sample order — two instances over the same source).
- `y = _labels_to_int_array(pool_aug.labels)`.
- `groups = np.asarray(pool_aug.groups)` if the dataset is grouped, else `None`.
- `official_test` = a held-out test `Dataset` (eval transform) if the source provides one,
  else `None` (the test set is then carved by `holdout_split`).

Import the new dataset class at the top of `train.py` (and `eval_clf.py`/`predict.py` if it
should be evaluable/visualizable there).

## Step 4: Verify & document

- Run the `/verify` skill (its data-free smoke test constructs models/losses; add a tiny
  check that the new dataset yields a `[1]` float label and, if grouped, a `groups` list).
- If a small sample of the data is available, run
  `DEVICE=cpu poetry run python -m src.modeling.train --data-type <key> --num-splits 2 --num-epochs 1`.
- Add the dataset to the `CLAUDE.md` package map and "How to run"; if you added a labeling
  rule or grouping, note it under the relevant invariant.
- Add any new third-party dependency with `poetry add <pkg>` (or `poetry add --group dev
  <pkg>` for tooling) so it lands in `[project.dependencies]` and `poetry.lock`.

Do not commit data or weights (`data/`, `models/` are gitignored and hook-blocked). Hand off
to `/commit-push` when ready.
