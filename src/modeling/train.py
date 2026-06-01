import copy
import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from loguru import logger
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, KFold,
                                     ShuffleSplit, StratifiedGroupKFold,
                                     StratifiedKFold, StratifiedShuffleSplit)
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModel

from src.config import (ALTERNATIVE_OOD_CATEGORIES, BASE_MODEL_NAME,
                        BATCH_SIZE, BRATS_TRAIN_DATA_DIR,
                        BRATS_TRAIN_SURVIVAL_INFO_PATH, DROPOUT_RATE,
                        IMAGE_SIZE, LR, MASK_RATIO, MIM_WEIGHT, MMD_WEIGHT,
                        MODELS_DIR, N_SPLITS, NUM_EPOCHS, OOD_CATEGORIES,
                        PATIENCE, RANDOM_STATE, SEGMENTED_TEST_ANNOTATIONS_PATH,
                        SEGMENTED_TEST_DATA_DIR,
                        SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR,
                        SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR,
                        TEST_FRAC, TRACKING_URI, ULTRASOUND_DATA_DIR,
                        ULTRASOUND_LABELS_MAPPING, WEIGHT_DECAY)
from src.modeling.data_processing import (BRATSSliceDataset, ImageDataset,
                                          ImageDatasetBrats, ImageDatasetCOCO,
                                          collate_fn, get_masked_images,
                                          resolve_concrete)
from src.modeling.models import (MIMTransformer,
                                 MultiLabelClassificationTransformer,
                                 MultiTaskTransformer)
from src.modeling.utils import (DEVICE, EVAL_TRANSFORM, TRAIN_TRANSFORM,
                                MultiTaskLoss, compute_mmd, confidence_interval,
                                get_classification_loss_function,
                                get_regression_loss_function)

app = typer.Typer(pretty_exceptions_show_locals=False)

CLASSIFICATION_TASKS = ('classification', 'multi-task')
METRIC_KEYS = ('loss', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc_score', 'pr_auc')


def get_model(base_model_name, num_classes, training_task, pre_trained_model_path=None):
    base_model = AutoModel.from_pretrained(
        base_model_name, add_pooling_layer=False, attn_implementation='eager'
    )
    if pre_trained_model_path:
        base_model.load_state_dict(torch.load(pre_trained_model_path, map_location=DEVICE))

    if training_task == 'mim':
        model = MIMTransformer(
            base_model, dropout_rate=DROPOUT_RATE
        )
    elif training_task == 'classification':
        model = MultiLabelClassificationTransformer(
            base_model, num_classes=num_classes, dropout_rate=DROPOUT_RATE
        )
    elif training_task == 'multi-task':
        model = MultiTaskTransformer(
            base_model, image_size=IMAGE_SIZE, num_classes=num_classes, dropout_rate=DROPOUT_RATE
        )
    else:
        raise ValueError(f'Invalid training task: {training_task}')

    return model


def get_loss_function(training_task, regression_loss_type=None, classification_loss=None, pos_weight=None):
    if training_task == 'mim':
        return get_regression_loss_function(regression_loss_type)
    elif training_task == 'classification':
        return get_classification_loss_function(classification_loss, pos_weight=pos_weight)
    elif training_task == 'multi-task':
        regression_loss = get_regression_loss_function(regression_loss_type)
        classification_loss = get_classification_loss_function(classification_loss, pos_weight=pos_weight)
        return MultiTaskLoss(regression_loss, classification_loss, MIM_WEIGHT)
    else:
        raise ValueError(f'Invalid training task: {training_task}')


# ---------------------------------------------------------------------------
# Out-of-domain / MMD helpers (experimental, off by default).
#
# NOTE (methodological limitation): the OOD batch is drawn from the SAME training
# dataset (alternative-OOD dental categories, or a random normal/disease batch), so this
# is a within-dataset feature-alignment regulariser rather than adaptation to a separate
# target domain. There is no dedicated OOD test set to validate that it improves
# robustness; treat results from --apply-mmd as exploratory.
# ---------------------------------------------------------------------------
def build_mmd_pools(dataset):
    """Precompute candidate index pools for sampling an OOD batch (once per fold)."""
    if len(dataset) == 0:
        raise ValueError("Empty dataset for OOD batch generation")

    concrete, _ = resolve_concrete(dataset, 0)
    full = list(range(len(dataset)))

    if isinstance(concrete, ImageDatasetCOCO):
        pool = []
        for i in full:
            ds, local_idx = resolve_concrete(dataset, i)
            if ds.get_categories(local_idx) & set(ALTERNATIVE_OOD_CATEGORIES):
                pool.append(i)
        return {'ood': pool or full}

    if isinstance(concrete, (ImageDataset, ImageDatasetBrats, BRATSSliceDataset)):
        # Binary labels are [0.] (negative) / [1.] (positive).
        normal, disease = [], []
        for i in full:
            ds, local_idx = resolve_concrete(dataset, i)
            (disease if int(ds.labels[local_idx][0]) == 1 else normal).append(i)
        return {'normal': normal or full, 'disease': disease or full}

    raise ValueError(f"Unsupported dataset {type(concrete)} for OOD batch generation")


def get_mmd_images(dataset, pools, batch_size=BATCH_SIZE):
    """Sample a batch of out-of-domain images from precomputed pools (see build_mmd_pools)."""
    if 'ood' in pools:
        indices = pools['ood']
    else:
        indices = pools['disease'] if random.choice([False, True]) else pools['normal']

    ood_loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return next(iter(ood_loader))['pixel_values'].to(DEVICE)


def forward_batch(model, criterion, batch, training_task, apply_mmd=False, mmd_weight=MMD_WEIGHT, mmd_images=None):
    images = batch['pixel_values'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)

    if training_task == 'mim':
        masked_images, mask = get_masked_images(images, model.base_model.config.patch_size)
        mim_output = model(masked_images)
        class_output = None
        # Reconstruct only the masked pixels, so the model cannot copy visible patches.
        loss = criterion(mim_output[~mask], images[~mask])
    elif training_task == 'classification':
        class_output = model(images)
        loss = criterion(class_output, labels.float())
    elif training_task == 'multi-task':
        masked_images, mask = get_masked_images(images, model.base_model.config.patch_size)
        mim_output, class_output = model(images, masked_images)
        loss = criterion(mim_output, class_output, images, labels.float(), mask)
    else:
        raise ValueError(f'Invalid training task: {training_task}')

    if apply_mmd:
        # Align in-domain and OOD distributions in the encoder's feature space (CLS token)
        # so the MMD penalty backpropagates through base_model (see module note above).
        in_domain_feats = model.base_model(images).last_hidden_state[:, 0, :]
        ood_feats = model.base_model(mmd_images).last_hidden_state[:, 0, :]
        loss = loss + mmd_weight * compute_mmd(in_domain_feats, ood_feats)

    return loss, class_output


def select_threshold(y_true, y_prob):
    """Pick the decision threshold that maximises F1 on the validation predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def compute_binary_metrics(y_true, y_prob, threshold=None):
    """Binary classification metrics on the positive class. Returns (metrics, threshold).

    PR-AUC (average precision) is the headline metric for the imbalanced caries task;
    ROC-AUC is reported alongside. Point metrics use a tuned (or supplied) threshold
    instead of a fixed 0.5 operating point.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_prob = np.asarray(y_prob).ravel()
    both_classes = len(np.unique(y_true)) > 1

    roc = roc_auc_score(y_true, y_prob) if both_classes else float('nan')
    pr_auc = average_precision_score(y_true, y_prob) if both_classes else float('nan')

    if threshold is None:
        threshold = select_threshold(y_true, y_prob) if both_classes else 0.5
    preds = (y_prob >= threshold).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds, zero_division=0),
        'recall': recall_score(y_true, preds, zero_division=0),
        'f1_score': f1_score(y_true, preds, zero_division=0),
        'roc_auc_score': roc,
        'pr_auc': pr_auc,
    }
    return metrics, float(threshold)


def collect_predictions(model, loader, training_task):
    """Run the model over a loader and return (probabilities, labels) as numpy arrays."""
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['pixel_values'].to(DEVICE)
            if training_task == 'multi-task':
                masked_images, _ = get_masked_images(images, model.base_model.config.patch_size)
                _, class_output = model(images, masked_images)
            else:
                class_output = model(images)
            probs.extend(torch.sigmoid(class_output).cpu().numpy())
            labels.extend(batch['labels'].numpy())
    return np.array(probs), np.array(labels)


def train_model(
        model, training_task, apply_mmd, device, train_loader, val_loader, criterion, optimizer, num_epochs, split_num
):
    model.to(device)

    # Precompute the OOD index pools once per fold rather than rescanning every step.
    mmd_pools = build_mmd_pools(train_loader.dataset) if apply_mmd else None

    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None
    best_threshold = None
    best_state = copy.deepcopy(model.state_dict())  # in case the first epoch is the best

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc='Training', unit='batch'):
            mmd_images = get_mmd_images(train_loader.dataset, mmd_pools) if apply_mmd else None
            loss, _ = forward_batch(model, criterion, batch, training_task, apply_mmd, mmd_images=mmd_images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')
        mlflow.log_metric('train_loss', avg_train_loss, step=epoch)

        # Validation
        model.eval()
        val_loss = 0.0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', unit='batch'):
                loss, class_output = forward_batch(model, criterion, batch, training_task)
                val_loss += loss.item()
                if class_output is not None:
                    all_probs.extend(torch.sigmoid(class_output).cpu().numpy())
                    all_labels.extend(batch['labels'].cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        mlflow.log_metric('val_loss', avg_val_loss, step=epoch)

        metrics = {'loss': round(avg_val_loss, 4)}
        threshold = None
        if training_task in CLASSIFICATION_TASKS:
            clf_metrics, threshold = compute_binary_metrics(np.array(all_labels), np.array(all_probs))
            metrics.update({k: round(float(v), 4) for k, v in clf_metrics.items()})

        mlflow.log_metrics(
            {f'epoch_{key}': v for key, v in metrics.items() if v == v}, step=epoch  # skip NaN
        )

        # Treat the first epoch as an improvement so best_* is always populated even if the
        # validation loss is NaN (diverged run); otherwise NaN < inf would never trigger.
        if best_metrics is None or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_metrics = metrics
            best_threshold = threshold
            best_state = copy.deepcopy(model.state_dict())  # checkpoint the BEST epoch, not the last
        else:
            patience_counter += 1
            logger.info(f'Patience counter: {patience_counter}')
            if patience_counter >= PATIENCE:
                logger.info(f'Early stopping triggered for fold {split_num + 1}.')
                break

    # Restore the best-epoch weights so the returned/saved model matches best_metrics.
    model.load_state_dict(best_state)
    return best_metrics, best_threshold, model


# ---------------------------------------------------------------------------
# Dataset construction and split helpers.
# ---------------------------------------------------------------------------
def _labels_to_int_array(label_lists):
    return np.array([int(float(label[0])) for label in label_lists])


def build_datasets(data_type):
    """Build the cross-validation pool (two transform variants) plus an optional official test set.

    Returns ``(pool_aug, pool_eval, y, groups, official_test)`` where ``pool_aug`` applies
    training augmentation and ``pool_eval`` applies the deterministic eval transform over the
    SAME samples in the same order, so a sample receives augmentation iff it is used for
    training in a given fold. ``official_test`` is the provider's held-out test set (COCO) or
    ``None`` (the test set is then carved from the pool).
    """
    if data_type == 'ultrasound':
        pool_aug = ImageDataset(ULTRASOUND_DATA_DIR, ULTRASOUND_LABELS_MAPPING, TRAIN_TRANSFORM)
        pool_eval = ImageDataset(ULTRASOUND_DATA_DIR, ULTRASOUND_LABELS_MAPPING, EVAL_TRANSFORM)
        return pool_aug, pool_eval, _labels_to_int_array(pool_aug.labels), None, None

    if data_type == 'coco':
        train_aug = ImageDatasetCOCO(SEGMENTED_TRAIN_ANNOTATIONS_PATH, SEGMENTED_TRAIN_DATA_DIR, TRAIN_TRANSFORM)
        val_aug = ImageDatasetCOCO(SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR, TRAIN_TRANSFORM)
        train_eval = ImageDatasetCOCO(SEGMENTED_TRAIN_ANNOTATIONS_PATH, SEGMENTED_TRAIN_DATA_DIR, EVAL_TRANSFORM)
        val_eval = ImageDatasetCOCO(SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR, EVAL_TRANSFORM)
        official_test = ImageDatasetCOCO(SEGMENTED_TEST_ANNOTATIONS_PATH, SEGMENTED_TEST_DATA_DIR, EVAL_TRANSFORM)
        pool_aug = ConcatDataset([train_aug, val_aug])
        pool_eval = ConcatDataset([train_eval, val_eval])
        y = _labels_to_int_array(list(train_aug.labels) + list(val_aug.labels))
        return pool_aug, pool_eval, y, None, official_test

    if data_type == 'brats':
        pool_aug = ImageDatasetBrats(BRATS_TRAIN_DATA_DIR, BRATS_TRAIN_SURVIVAL_INFO_PATH, TRAIN_TRANSFORM)
        pool_eval = ImageDatasetBrats(BRATS_TRAIN_DATA_DIR, BRATS_TRAIN_SURVIVAL_INFO_PATH, EVAL_TRANSFORM)
        return pool_aug, pool_eval, _labels_to_int_array(pool_aug.labels), None, None

    if data_type == 'brats_slice':
        pool_aug = BRATSSliceDataset(BRATS_TRAIN_DATA_DIR, transform=TRAIN_TRANSFORM)
        pool_eval = BRATSSliceDataset(BRATS_TRAIN_DATA_DIR, transform=EVAL_TRANSFORM)
        # Group by scan so slices of the same patient never straddle a split.
        return pool_aug, pool_eval, _labels_to_int_array(pool_aug.labels), np.asarray(pool_aug.groups), None

    raise ValueError(f'Invalid data type: {data_type}')


def holdout_split(indices, y, groups, test_frac, seed):
    """Carve a stratified (and group-aware, if applicable) held-out test set from ``indices``."""
    indices = np.asarray(indices)
    y_sub = np.asarray(y)[indices]
    if groups is not None:
        g_sub = np.asarray(groups)[indices]
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr, te = next(splitter.split(indices, y_sub, g_sub))
    elif len(np.unique(y_sub)) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr, te = next(splitter.split(indices, y_sub))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        tr, te = next(splitter.split(indices))
    return indices[tr], indices[te]


def kfold_splits(indices, y, groups, n_splits, seed):
    """Disjoint k-fold splits (stratified and/or grouped), returning absolute index arrays."""
    indices = np.asarray(indices)
    y_sub = np.asarray(y)[indices]
    n_classes = len(np.unique(y_sub))

    if groups is not None:
        g_sub = np.asarray(groups)[indices]
        if n_classes >= 2:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        else:
            splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(indices, y_sub, g_sub)
    elif n_classes >= 2:
        iterator = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(indices, y_sub)
    else:
        iterator = KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(indices)

    return [(indices[tr], indices[va]) for tr, va in iterator]


def filter_ood_indices(pool, indices):
    """Drop indices whose COCO image contains an out-of-domain category."""
    kept = []
    for i in indices:
        ds, local_idx = resolve_concrete(pool, int(i))
        cats = ds.get_categories(local_idx) if isinstance(ds, ImageDatasetCOCO) else set()
        if not (cats & set(OOD_CATEGORIES)):
            kept.append(int(i))
    dropped = len(indices) - len(kept)
    if dropped:
        logger.info(f'filter_ood: dropped {dropped}/{len(indices)} OOD-category training images')
    return np.asarray(kept)


@app.command()
def main(
        base_model_name: str = BASE_MODEL_NAME,
        pre_trained_model_run_id: str = None,
        output_dir: Path = MODELS_DIR,
        training_task: str = 'multi-task',
        data_type: str = 'coco',
        regression_loss_type: str = 'MSE',
        classification_loss_type: str = 'BCEWithLogits',
        apply_mmd: bool = False,
        filter_ood: bool = False,
        batch_size: int = BATCH_SIZE,
        num_splits: int = N_SPLITS,
        num_epochs: int = NUM_EPOCHS,
        test_frac: float = TEST_FRAC,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        num_workers: int = 0,
):
    """Train a ViT (MIM / classification / multi-task) with stratified k-fold cross-validation.

    The provider's COCO test split (or, for the other datasets, a freshly carved stratified
    held-out set) is kept untouched during training/selection and used only for a single final
    evaluation. Cross-validation uses disjoint folds; per-fold metrics are aggregated with
    t-based confidence intervals.
    """
    experiment_name = {
        'mim': 'MIM Experiment',
        'classification': 'Classification Experiment',
        'multi-task': 'Multi-Task Experiment',
    }.get(training_task)
    if experiment_name is None:
        raise ValueError(f'Invalid training task: {training_task}')

    if num_splits < 2:
        raise ValueError('num_splits must be >= 2 for k-fold cross-validation')

    # Seed all RNGs for reproducible splits, masking, and model initialisation.
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.cuda.manual_seed_all(RANDOM_STATE)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as main_run:
        logger.info(f'MLflow experiment ID: {experiment.experiment_id}')
        logger.info(f'MLflow run ID: {main_run.info.run_id}')

        artifacts_dir = output_dir / main_run.info.run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Build the CV pool (augmented + eval transform variants) and the held-out test set.
        pool_aug, pool_eval, y, groups, official_test = build_datasets(data_type)
        num_classes = 1  # single binary head

        all_indices = np.arange(len(pool_aug))
        if official_test is not None:
            trainval_idx = all_indices
            test_indices_in_pool = None
        else:
            trainval_idx, test_idx = holdout_split(all_indices, y, groups, test_frac, RANDOM_STATE)
            test_indices_in_pool = test_idx

        folds = kfold_splits(trainval_idx, y, groups, num_splits, RANDOM_STATE)

        # Class-imbalance weighting from the (training) pool positive rate.
        y_trainval = y[trainval_idx]
        n_pos = int(y_trainval.sum())
        n_neg = int(len(y_trainval) - n_pos)
        pos_weight = (
            torch.tensor([n_neg / n_pos], dtype=torch.float32, device=DEVICE)
            if training_task in CLASSIFICATION_TASKS and n_pos > 0 else None
        )

        mlflow.log_params({
            'base_model_name': base_model_name,
            'pre_trained_model_run_id': pre_trained_model_run_id,
            'training_task': training_task,
            'data_type': data_type,
            'regression_loss_type': regression_loss_type,
            'classification_loss_type': classification_loss_type,
            'apply_mmd': apply_mmd,
            'filter_ood': filter_ood,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'patience': PATIENCE,
            'cv_pool_size': len(trainval_idx),
            'test_size': len(official_test) if official_test is not None else len(test_indices_in_pool),
            'num_classes': num_classes,
            'event_rate': float(y_trainval.mean()),
            'pos_weight': float(pos_weight.item()) if pos_weight is not None else None,
            'test_frac': test_frac if official_test is None else 'official_split',
            'learning_rate': lr,
            'num_splits': num_splits,
            'weight_decay': weight_decay,
            'mim_weight': MIM_WEIGHT,
            'mask_ratio': MASK_RATIO,
            'num_workers': num_workers,
        })

        overall_metrics = {key: [] for key in METRIC_KEYS}
        fold_thresholds = []

        for split_num, (train_idx, val_idx) in enumerate(folds):
            if filter_ood:
                train_idx = filter_ood_indices(pool_aug, train_idx)
                if len(train_idx) == 0:
                    raise ValueError(
                        f'filter_ood removed every training sample in fold {split_num + 1}; '
                        f'OOD_CATEGORIES is too broad for this dataset'
                    )

            # Train folds use augmentation; the validation fold uses the deterministic transform.
            train_subset = Subset(pool_aug, list(train_idx))
            val_subset = Subset(pool_eval, list(val_idx))

            with mlflow.start_run(run_name=f'Fold_{split_num + 1}', nested=True) as split_run:
                logger.info(f'Fold {split_num + 1}/{num_splits}, Run ID: {split_run.info.run_id}')

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                                          collate_fn=collate_fn, num_workers=num_workers)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                                        collate_fn=collate_fn, num_workers=num_workers)

                pre_trained_model_path = (
                    output_dir / pre_trained_model_run_id / 'base_model.pth'
                    if pre_trained_model_run_id else None
                )

                model = get_model(base_model_name, num_classes, training_task, pre_trained_model_path)
                criterion = get_loss_function(training_task, regression_loss_type, classification_loss_type, pos_weight)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

                metrics, threshold, model = train_model(
                    model=model, training_task=training_task, apply_mmd=apply_mmd, device=DEVICE,
                    train_loader=train_loader, val_loader=val_loader, criterion=criterion,
                    optimizer=optimizer, num_epochs=num_epochs, split_num=split_num,
                )

                mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})  # skip NaN
                for key, value in metrics.items():
                    overall_metrics[key].append(value)
                fold_thresholds.append(threshold)

                fold_model_path = artifacts_dir / f'fold_{split_num}_model.pth'
                torch.save(model.state_dict(), fold_model_path)
                mlflow.log_artifact(fold_model_path)

        # Aggregate per-fold metrics with t-based confidence intervals (folds are disjoint).
        avg_metrics, conf_lower_metrics, conf_upper_metrics = {}, {}, {}
        for key, values in overall_metrics.items():
            values = [v for v in values if v == v]  # drop NaNs (e.g. degenerate-fold roc_auc)
            if not values:
                continue
            mean, lower, upper = confidence_interval(values)
            avg_metrics[f'avg_{key}'] = round(mean, 4)
            conf_lower_metrics[f'{key}_lower'] = round(lower, 4)
            conf_upper_metrics[f'{key}_upper'] = round(upper, 4)

        logger.info(f'Cross-validation metrics (mean): {avg_metrics}')
        mlflow.log_metrics(avg_metrics)
        mlflow.log_metrics(conf_lower_metrics)
        mlflow.log_metrics(conf_upper_metrics)

        # Select the BEST fold (lowest validation loss), not one near the average.
        best_fold = int(np.argmin(overall_metrics['loss']))
        best_threshold = fold_thresholds[best_fold]
        logger.info(f'Selected fold {best_fold + 1} (lowest validation loss) as the exported model')

        model = get_model(base_model_name, num_classes, training_task)
        model.load_state_dict(torch.load(artifacts_dir / f'fold_{best_fold}_model.pth', map_location=DEVICE))
        model.to(DEVICE)

        base_model_path = artifacts_dir / 'base_model.pth'
        model_path = artifacts_dir / 'model.pth'
        torch.save(model.base_model.state_dict(), base_model_path)
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(base_model_path)
        mlflow.log_artifact(model_path)
        if best_threshold is not None:
            mlflow.log_param('decision_threshold', round(best_threshold, 4))

        # Single, final evaluation on the untouched held-out test set (classification tasks only).
        if training_task in CLASSIFICATION_TASKS:
            test_subset = official_test if official_test is not None else Subset(pool_eval, list(test_indices_in_pool))
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            y_prob, y_true = collect_predictions(model, test_loader, training_task)
            test_metrics, _ = compute_binary_metrics(y_true, y_prob, threshold=best_threshold)
            test_metrics = {f'test_{k}': round(float(v), 4) for k, v in test_metrics.items()}
            logger.info(f'Held-out test metrics: {test_metrics}')
            mlflow.log_metrics(test_metrics)

        logger.info(f'Best base model saved at {base_model_path}')
        logger.info(f'Best model saved at {model_path}')
        logger.info('Cross-validation completed. All metrics logged.')


if __name__ == '__main__':
    app()
