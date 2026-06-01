---
name: verify
description: Smoke-check that the training/eval code actually runs end-to-end, catching runtime breakage (tensor shapes, masking, loss wiring, constructor signatures, sklearn split usage) that compile + lint cannot. Prefers a data-free synthetic forward/backward smoke test (no dataset/GPU needed); optionally runs a tiny real 1-epoch / 2-fold training if data is present. Use after non-trivial changes to models, train, data_processing, utils, or eval — especially since this repo has no test suite.
---

# Verify (med-vision-transformers)

This repo has **no test suite**, so a change can compile, lint, and still crash at runtime
(a shape mismatch, a wrong constructor signature, a masking/loss wiring bug, a sklearn
splitter misuse). This skill exercises the real code paths to catch that.

## Arguments

`$ARGUMENTS` — optional: `--real` to also attempt a tiny real training run if data exists;
a task/data_type to scope. Default = the data-free synthetic smoke test (fast, no deps on
data or GPU beyond the installed packages).

## Flow

### Step 1: Compile

```bash
python -m compileall -q src && echo "compile: ok"
```

### Step 2: Data-free synthetic smoke test (default)

Drive the actual model/loss/masking code on **random tensors of the correct shape**, with no
dataset, on CPU. Construct each transformer via the real factory and run a forward + backward
for each task, plus the masking and metric helpers. Write a short throwaway script and run
it, e.g.:

```bash
DEVICE=cpu python - <<'PY'
import torch
from transformers import AutoModel
from src.config import BASE_MODEL_NAME, IMAGE_SIZE
from src.modeling.models import (MIMTransformer, MultiLabelClassificationTransformer,
                                  MultiTaskTransformer)
from src.modeling.data_processing import get_masked_images
from src.modeling.utils import MultiTaskLoss, get_regression_loss_function, get_classification_loss_function

base = AutoModel.from_pretrained(BASE_MODEL_NAME, add_pooling_layer=False, attn_implementation='eager')
ps = base.config.patch_size
x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
y = torch.tensor([[0.], [1.]])

# MIM: masked-only reconstruction in normalized space
mim = MIMTransformer(base, dropout_rate=0.1)
masked, mask = get_masked_images(x, ps)
out = mim(masked)
assert out.shape == x.shape, out.shape
loss = get_regression_loss_function('MSE')(out[~mask], x[~mask]); loss.backward()
assert float(mask.float().mean()) < 1.0  # something is actually masked
print('MIM ok', out.shape, 'masked_frac', round(1 - float(mask.float().mean()), 3))

# Classification: single logit + BCE
clf = MultiLabelClassificationTransformer(base, num_classes=1, dropout_rate=0.1)
logit = clf(x); assert logit.shape == (2, 1), logit.shape
get_classification_loss_function('BCEWithLogits', pos_weight=torch.tensor([1.0]))(logit, y).backward()
print('CLF ok', logit.shape)

# Multi-task: masked MIM + classification
mt = MultiTaskTransformer(base, image_size=IMAGE_SIZE, num_classes=1, dropout_rate=0.1)
masked, mask = get_masked_images(x, ps)
mim_out, cls_out = mt(x, masked)
MultiTaskLoss(get_regression_loss_function('MSE'),
              get_classification_loss_function('BCEWithLogits'), 0.1)(mim_out, cls_out, x, y, mask).backward()
print('MULTITASK ok', mim_out.shape, cls_out.shape)
PY
```

Also smoke the split/metric helpers without data:

```bash
python - <<'PY'
import numpy as np
from src.modeling.train import kfold_splits, holdout_split, compute_binary_metrics, select_threshold
from src.modeling.utils import confidence_interval
y = np.array([0,1]*20); idx = np.arange(len(y))
tr, te = holdout_split(idx, y, None, 0.2, 214)
folds = kfold_splits(tr, y, None, 5, 214)
assert all(set(a).isdisjoint(b) for a, b in folds)              # disjoint train/val
assert all(set(te).isdisjoint(set(a)|set(b)) for a, b in folds)  # test never in a fold
m, _ = compute_binary_metrics(np.array([0,0,1,1]), np.array([.1,.2,.7,.9]))
print('splits ok; pr_auc', round(m['pr_auc'], 3), 'ci', confidence_interval([0.8,0.82,0.79]))
PY
```

Tailor the script to what the diff touched (e.g. construct a new model, exercise a new
loss). Report any traceback verbatim as a finding.

### Step 3: Optional real run (`--real`, only if data exists)

If the dataset directory exists, run a tiny real training to exercise the dataset/dataloader/
CV path too (keep it fast: 1 epoch, 2 folds):

```bash
DEVICE=cpu python -m src.modeling.train --training-task multi-task --data-type coco \
  --num-splits 2 --num-epochs 1 --batch-size 4
```

Watch for runtime errors; you do not need good metrics, just a clean run to the "Best model
saved" log. If data is absent, say so and rely on Step 2.

### Step 4: Report

State clearly what was exercised (synthetic forward/backward for which tasks, the split/
metric helpers, and whether a real run was attempted), and PASS/FAIL with any traceback.
Be explicit that the synthetic smoke test validates shapes/signatures/wiring but **not**
data loading or training quality. Do not commit anything.
