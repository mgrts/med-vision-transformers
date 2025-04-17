import random
from pathlib import Path

import mlflow
import numpy as np
import torch
import typer
from loguru import logger
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm
from transformers import AutoModel

from src.config import (ALTERNATIVE_OOD_CATEGORIES, BASE_MODEL_NAME,
                        BATCH_SIZE, BRATS_TRAIN_DATA_DIR,
                        BRATS_TRAIN_SURVIVAL_INFO_PATH, DROPOUT_RATE,
                        IMAGE_SIZE, LR, MASK_RATIO, MIM_WEIGHT, MMD_WEIGHT,
                        MODELS_DIR, NUM_EPOCHS, OOD_CATEGORIES, PATIENCE,
                        SEGMENTED_TEST_ANNOTATIONS_PATH,
                        SEGMENTED_TEST_DATA_DIR,
                        SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR,
                        SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR,
                        TRACKING_URI, ULTRASOUND_DATA_DIR,
                        ULTRASOUND_LABELS_MAPPING, WEIGHT_DECAY)
from src.modeling.data_processing import (BRATSSliceDataset, ImageDataset,
                                          ImageDatasetBrats, ImageDatasetCOCO,
                                          collate_fn, create_mask)
from src.modeling.models import (MIMTransformer,
                                 MultiLabelClassificationTransformer,
                                 MultiTaskTransformer)
from src.modeling.utils import (DEVICE, EVAL_TRANSFORM, TRAIN_TRANSFORM,
                                MultiTaskLoss,
                                get_classification_loss_function,
                                get_regression_loss_function)

app = typer.Typer(pretty_exceptions_show_locals=False)


def get_model(base_model_name, num_classes, training_task, pre_trained_model_path=None):
    base_model = AutoModel.from_pretrained(
        base_model_name, add_pooling_layer=False, attn_implementation='eager'
    )
    if pre_trained_model_path:
        base_model.load_state_dict(torch.load(pre_trained_model_path))

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


def get_loss_function(training_task, regression_loss_type=None, classification_loss=None):
    if training_task == 'mim':
        return get_regression_loss_function(regression_loss_type)
    elif training_task == 'classification':
        return get_classification_loss_function(classification_loss)
    elif training_task == 'multi-task':
        regression_loss = get_regression_loss_function(regression_loss_type)
        classification_loss = get_classification_loss_function(classification_loss)
        return MultiTaskLoss(regression_loss, classification_loss, MIM_WEIGHT)
    else:
        raise ValueError(f'Invalid training task: {training_task}')


def get_mmd_images(dataset, batch_size=BATCH_SIZE):
    """
    Generate an MMD batch based on the specified data type.

    Args:
        dataset (Dataset): The main dataset.
        data_type (str): Type of the dataset, e.g., 'real' or 'ultrasound'.
        batch_size (int): Number of samples in the MMD batch.

    Returns:
        torch.Tensor: A batch of images to be used for MMD calculation.
    """
    if isinstance(dataset, ImageDatasetCOCO):
        # Filter for OOD categories (Alternative OOD categories)
        ood_alt_indices = [i for i, sample in enumerate(dataset) if sample['category'] in ALTERNATIVE_OOD_CATEGORIES]
        ood_alt_dataset = Subset(dataset, ood_alt_indices)

        # Create a DataLoader and get a batch from it
        ood_alt_loader = DataLoader(ood_alt_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        ood_batch = next(iter(ood_alt_loader))
        return ood_batch['pixel_values'].to(DEVICE)

    elif isinstance(dataset, ImageDataset) or isinstance(dataset, ImageDatasetBrats) or isinstance(dataset, BRATSSliceDataset):
        # Separate indices for normal and disease classes based on labels
        normal_indices = [i for i, label in enumerate(dataset.labels) if label == 0]
        disease_indices = [i for i, label in enumerate(dataset.labels) if label == 1]

        # Randomly choose normal or disease for OOD batch
        if random.choice(['normal', 'disease']) == 'normal':
            ood_dataset = Subset(dataset, normal_indices)
        else:
            ood_dataset = Subset(dataset, disease_indices)

        # Create DataLoader and retrieve a batch
        ood_loader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        ood_batch = next(iter(ood_loader))
        return ood_batch['pixel_values'].to(DEVICE)

    else:
        raise ValueError(f"Unsupported dataset {type(dataset)} for OOD batch generation")


def compute_mmd(x, y, kernel=torch.nn.functional.pairwise_distance):
    """Compute the MMD loss between two sets of samples, x and y."""
    xx = kernel(x.unsqueeze(1), x.unsqueeze(0)).mean()
    yy = kernel(y.unsqueeze(1), y.unsqueeze(0)).mean()
    xy = kernel(x.unsqueeze(1), y.unsqueeze(0)).mean()
    mmd_loss = xx + yy - 2 * xy
    return mmd_loss


def get_masked_images(images, mask_ratio=MASK_RATIO):
    batch_size, channels, height, width = images.shape
    mask = create_mask(batch_size, height, width, mask_ratio).to(images.device)
    masked_images = images.clone()
    masked_images[~mask] = 1.0
    return masked_images


def forward_batch(model, criterion, batch, training_task, apply_mmd=False, mmd_weight=MMD_WEIGHT, mmd_images=None):
    images = batch['pixel_values'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)

    if training_task == 'mim':
        masked_images = get_masked_images(images)
        mim_output = model(masked_images)
        class_output = None
        loss = criterion(mim_output, images)
    elif training_task == 'classification':
        class_output = model(images)
        loss = criterion(class_output, labels.float())
    elif training_task == 'multi-task':
        masked_images = get_masked_images(images)
        mim_output, class_output = model(images, masked_images)
        loss = criterion(mim_output, class_output, images, labels.float())
    else:
        raise ValueError(f'Invalid training task: {training_task}')

    if apply_mmd:
        mmd_loss = compute_mmd(images, mmd_images)
        loss += mmd_weight * mmd_loss

    return loss, class_output


def train_model(
        model, training_task, apply_mmd, device, train_loader, val_loader, criterion, optimizer, num_epochs, split_num
):
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    best_metrics = None

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        # Training
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc='Training', unit='batch'):
            mmd_images = get_mmd_images(train_loader.dataset) if apply_mmd else None
            loss, class_output = forward_batch(model, criterion, batch, training_task, apply_mmd, mmd_images=mmd_images)

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
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation', unit='batch'):
                loss, class_output = forward_batch(model, criterion, batch, training_task)
                labels = batch['labels']

                val_loss += loss.item()

                if class_output is not None:
                    probs = torch.sigmoid(class_output).cpu().numpy()
                    labels = labels.cpu().numpy()
                    all_probs.extend(probs)
                    all_labels.extend(labels)

        avg_val_loss = val_loss / len(val_loader)

        logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')
        mlflow.log_metric('val_loss', avg_val_loss, step=epoch)

        metrics = {'loss': round(avg_val_loss, 4)}
        if training_task in ('classification', 'multi-task'):
            all_probs = np.array(all_probs)
            all_labels = np.array(all_labels)

            accuracy = accuracy_score(all_labels, np.round(all_probs))
            precision = precision_score(all_labels, np.round(all_probs), average='macro', zero_division=0)
            recall = recall_score(all_labels, np.round(all_probs), average='macro', zero_division=0)
            f1 = f1_score(all_labels, np.round(all_probs), average='macro', zero_division=0)
            roc_auc = roc_auc_score(all_labels, all_probs, average='macro')

            metrics['accuracy'] = round(accuracy, 4)
            metrics['precision'] = round(precision, 4)
            metrics['recall'] = round(recall, 4)
            metrics['f1_score'] = round(f1, 4)
            metrics['roc_auc_score'] = round(roc_auc, 4)

        mlflow.log_metrics({f'epoch_{key}': value for key, value in metrics.items()}, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_metrics = metrics
        else:
            patience_counter += 1
            logger.info(f'Patience counter: {patience_counter}')
            if patience_counter >= PATIENCE:
                logger.info(f'Early stopping triggered for Split {split_num + 1}.')
                break

    return best_metrics, model


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
        num_splits: int = 15,
        num_epochs: int = NUM_EPOCHS,
        test_frac: float = 0.2,
        lr: float = LR,
        weight_decay: float = WEIGHT_DECAY,
        num_workers: int = 0,
):
    """
    Train Visual Transformer for classification and MIM with cross-validation.
    """
    if training_task == 'mim':
        experiment_name = 'MIM Experiment'
    elif training_task == 'classification':
        experiment_name = 'Classification Experiment'
    elif training_task == 'multi-task':
        experiment_name = 'Multi-Task Experiment'
    else:
        raise ValueError(f'Invalid training task: {training_task}')

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    with mlflow.start_run(experiment_id=experiment.experiment_id) as main_run:
        logger.info(f'MLflow experiment ID: {experiment.experiment_id}')
        logger.info(f'MLflow run ID: {main_run.info.run_id}')

        artifacts_dir = output_dir / main_run.info.run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        if data_type == 'ultrasound':
            dataset = ImageDataset(
                image_dir=ULTRASOUND_DATA_DIR,
                label_mapping=ULTRASOUND_LABELS_MAPPING,
                transform=TRAIN_TRANSFORM
            )
            labels = torch.tensor(dataset.labels, dtype=torch.long)
        elif data_type == 'coco':
            train_dataset = ImageDatasetCOCO(
                annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                image_dir=SEGMENTED_TRAIN_DATA_DIR,
                transform=TRAIN_TRANSFORM,
            )
            val_dataset = ImageDatasetCOCO(
                annotation_file=SEGMENTED_VAL_ANNOTATIONS_PATH,
                image_dir=SEGMENTED_VAL_DATA_DIR,
                transform=EVAL_TRANSFORM,
            )
            test_dataset = ImageDatasetCOCO(
                annotation_file=SEGMENTED_TEST_ANNOTATIONS_PATH,
                image_dir=SEGMENTED_TEST_DATA_DIR,
                transform=EVAL_TRANSFORM,
            )
            dataset = train_dataset + val_dataset + test_dataset
            labels = torch.cat([
                torch.stack(train_dataset.labels),
                torch.stack(val_dataset.labels),
                torch.stack(test_dataset.labels)
            ], dim=0)
        elif data_type == 'brats':
            dataset = ImageDatasetBrats(
                image_dir=BRATS_TRAIN_DATA_DIR,
                info_path=BRATS_TRAIN_SURVIVAL_INFO_PATH,
                transform=TRAIN_TRANSFORM
            )
            labels = torch.tensor(dataset.labels, dtype=torch.long)
        elif data_type == 'brats_slice':
            dataset = BRATSSliceDataset(
                image_dir=BRATS_TRAIN_DATA_DIR,
                transform=TRAIN_TRANSFORM,
            )
            labels = torch.tensor(dataset.labels, dtype=torch.long)
        else:
            raise ValueError(f'Invalid data type: {data_type}')

        num_classes = labels.shape[1]

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
            'num_epochs': NUM_EPOCHS,
            'patience': PATIENCE,
            'dataset_size': len(dataset),
            'num_classes': num_classes,
            'event_rate': np.array(labels).mean(),
            'test_frac': test_frac,
            'learning_rate': LR,
            'num_splits': num_splits,
            'weight_decay': WEIGHT_DECAY,
            'mim_weight': MIM_WEIGHT,
            'mask_ratio': MASK_RATIO,
            'num_workers': num_workers,
        })

        overall_metrics = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'roc_auc_score': []
        }

        for split_num in range(num_splits):
            train_size = int((1 - test_frac) * len(dataset))
            train_size = train_size // 10
            print(f"train_size: {train_size}")
            val_size = len(dataset) - train_size
            train_subset, val_subset = random_split(dataset, [train_size, val_size])

            if filter_ood:
                train_subset = Subset(
                    train_subset,
                    [i for i, sample in enumerate(train_subset) if sample['category'] not in OOD_CATEGORIES]
                )

            with mlflow.start_run(run_name=f'Split_{split_num + 1}', nested=True) as split_run:
                logger.info(f'Split {split_num + 1}/{num_splits}, Run ID: {split_run.info.run_id}')

                train_loader = DataLoader(
                    dataset=train_subset,
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=num_workers
                )
                val_loader = DataLoader(
                    dataset=val_subset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=num_workers
                )

                if pre_trained_model_run_id:
                    pre_trained_model_path = output_dir / pre_trained_model_run_id / 'base_model.pth'
                else:
                    pre_trained_model_path = None

                model = get_model(base_model_name, num_classes, training_task, pre_trained_model_path)
                criterion = get_loss_function(training_task, regression_loss_type, classification_loss_type)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

                metrics, model = train_model(
                    model=model,
                    training_task=training_task,
                    apply_mmd=apply_mmd,
                    device=DEVICE,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    split_num=split_num
                )
                # Log the best metrics for the split
                mlflow.log_metrics(metrics)

                for key, value in metrics.items():
                    overall_metrics[key].append(value)

                split_model_path = artifacts_dir / f'split_{split_num}_model.pth'
                torch.save(model.state_dict(), split_model_path)
                mlflow.log_artifact(split_model_path)

        # Compute and log average metrics across all splits
        avg_metrics = dict()
        conf_lower_metrics = dict()
        conf_upper_metrics = dict()
        for key, values in overall_metrics.items():
            avg_metrics[f'avg_{key}'] = round(float(np.mean(values)), 4)
            sigma = np.std(values, ddof=1)
            conf_lower_metrics[f'{key}_lower'] = round(avg_metrics[f'avg_{key}'] - 2 * sigma, 4)
            conf_upper_metrics[f'{key}_upper'] = round(avg_metrics[f'avg_{key}'] + 2 * sigma, 4)

        logger.info(f'Average metrics across all splits: {avg_metrics}')

        mlflow.log_metrics(avg_metrics)
        mlflow.log_metrics(conf_lower_metrics)
        mlflow.log_metrics(conf_upper_metrics)

        # select model closest to the average loss
        avg_loss = avg_metrics['avg_loss']
        best_diff = float('inf')
        for i, loss_value in enumerate(overall_metrics['loss']):
            if abs(loss_value - avg_loss) < best_diff:
                best_diff = abs(loss_value - avg_loss)
                best_split_num = i

        best_model_path = artifacts_dir / f'split_{best_split_num}_model.pth'
        model.load_state_dict(torch.load(best_model_path))
        base_model = model.base_model

        base_model_path = artifacts_dir / 'base_model.pth'
        model_path = artifacts_dir / 'model.pth'

        torch.save(base_model.state_dict(), base_model_path)
        torch.save(model.state_dict(), model_path)

        mlflow.log_artifact(base_model_path)
        mlflow.log_artifact(model_path)

        logger.info(f'Best base model saved at {base_model_path}')
        logger.info(f'Best model saved at {model_path}')
        logger.info('Cross-validation completed. All metrics logged.')


if __name__ == '__main__':
    app()
