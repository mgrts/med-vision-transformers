import mlflow
import numpy as np
import torch
import typer
from loguru import logger
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.config import (BASE_MODEL_NAME, BATCH_SIZE, IMAGE_SIZE, MASK_RATIO,
                        MODELS_DIR, SEGMENTED_TEST_ANNOTATIONS_PATH,
                        SEGMENTED_TEST_DATA_DIR,
                        SEGMENTED_TRAIN_ANNOTATIONS_PATH,
                        SEGMENTED_TRAIN_DATA_DIR,
                        SEGMENTED_VAL_ANNOTATIONS_PATH, SEGMENTED_VAL_DATA_DIR,
                        TRACKING_URI)
from src.modeling.data_processing import (ImageDatasetCOCO, collate_fn,
                                          create_mask)
from src.modeling.models import (MultiLabelClassificationTransformer,
                                 MultiTaskTransformer)
from src.modeling.utils import DEVICE, TRAIN_TRANSFORM

app = typer.Typer(pretty_exceptions_show_locals=False)

RUN_TYPE = 'multi-task l1'


def compute_2sigma_confidence_interval(data):
    """
    Compute the confidence interval for a given data array using 2-sigma (two standard deviations).
    """
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    margin_of_error = 2 * std_dev  # 2-sigma is approximately 95% confidence interval

    return mean, mean - margin_of_error, mean + margin_of_error


def get_masked_images(images, mask_ratio=MASK_RATIO):
    batch_size, channels, height, width = images.shape
    mask = create_mask(batch_size, height, width, mask_ratio).to(images.device)
    masked_images = images.clone()
    masked_images[~mask] = 1.0
    return masked_images


def forward_batch(model, batch, training_task):
    images = batch['pixel_values'].to(DEVICE)
    labels = batch['labels'].to(DEVICE)

    if training_task == 'classification':
        class_output = model(images)
    elif training_task == 'multi-task':
        masked_images = get_masked_images(images)
        mim_output, class_output = model(images, masked_images)
    else:
        raise ValueError(f'Invalid training task: {training_task}')

    return class_output, labels


def evaluate_model(model, dataloader, device, training_task):
    """
    Evaluate the model and calculate classification metrics.
    """
    probs = []
    labels_all = []
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            class_output, labels = forward_batch(model, batch, training_task)
            probs.extend(torch.sigmoid(class_output).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    return np.array(probs), np.array(labels_all)


def get_parent_run_id(run_type):
    if run_type == 'no pre-training':
        return 'cb34f67f2ed148b78133d3344a17abae'
    elif run_type == 'pre-trained l1':
        return '643620bc79024dcca5940284c570a8c7'
    elif run_type == 'pre-trained mse':
        return '168b16c539554ba29404180d42b22e12'
    elif run_type == 'multi-task mse':
        return 'b7a8f355026742dbaf456da8e2fc7cd1'
    elif run_type == 'multi-task l1':
        return '3f6408c6505f4401a561f8365f5b2bce'
    else:
        raise ValueError(f'Unsupported run type: {run_type}')


@app.command()
def main(
        parent_run_id: str = get_parent_run_id(RUN_TYPE),
        dataset_type: str = 'coco',
        batch_size: int = BATCH_SIZE
):
    """
    Evaluate a model for N splits from the dataset using a parent MLflow run ID.
    Save classification metrics for each split and calculate average ROC AUC and its confidence interval.

    Args:
        parent_run_id (str): MLflow parent run ID for nested runs.
        n_splits (int): Number of splits for cross-validation.
        dataset_type (str): Dataset type ('coco').
        batch_size (int): Batch size for evaluation.
    """
    logger.info(f'Evaluating model from parent run ID: {parent_run_id}')

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('evaluate_clf')
    experiment = mlflow.get_experiment_by_name('evaluate_clf')

    client = mlflow.tracking.MlflowClient()

    # Fetch child runs
    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
    )

    # Prepare dataset
    if dataset_type == 'coco':
        train_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TRAIN_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TRAIN_DATA_DIR,
            transform=TRAIN_TRANSFORM,
        )
        val_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_VAL_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_VAL_DATA_DIR,
            transform=TRAIN_TRANSFORM,
        )
        test_dataset = ImageDatasetCOCO(
            annotation_file=SEGMENTED_TEST_ANNOTATIONS_PATH,
            image_dir=SEGMENTED_TEST_DATA_DIR,
            transform=TRAIN_TRANSFORM,
        )
        dataset = train_dataset + val_dataset + test_dataset
        labels = torch.cat([
            torch.stack(train_dataset.labels),
            torch.stack(val_dataset.labels),
            torch.stack(test_dataset.labels)
        ], dim=0)
    else:
        raise ValueError(f'Unsupported dataset type: {dataset_type}')

    num_classes = labels.shape[1]

    split_metrics = []
    roc_auc_scores = []

    for run in child_runs:
        run_artifacts = client.list_artifacts(run.info.run_id)
        run_name = run.data.tags['mlflow.runName']

        logger.info(f'Evaluating Run {run_name}')

        model_uri = MODELS_DIR / parent_run_id / run_artifacts[0].path

        # Load model
        base_model = AutoModel.from_pretrained(
            BASE_MODEL_NAME, add_pooling_layer=False, attn_implementation='eager'
        )
        if RUN_TYPE in ('no pre-training', 'pre-trained l1', 'pre-trained mse'):
            model = MultiLabelClassificationTransformer(base_model, num_classes=num_classes)
        else:
            model = MultiTaskTransformer(base_model, image_size=IMAGE_SIZE, num_classes=num_classes)

        model.load_state_dict(torch.load(model_uri))
        model.eval()

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        if RUN_TYPE in ('no pre-training', 'pre-trained l1', 'pre-trained mse'):
            training_task = 'classification'
        else:
            training_task = 'multi-task'

        # Evaluate the model
        y_pred, y_true = evaluate_model(model, dataloader, DEVICE, training_task)

        # Calculate metrics
        y_pred_rounded = np.round(y_pred)
        accuracy = accuracy_score(y_true, y_pred_rounded)
        precision = precision_score(y_true, y_pred_rounded, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_rounded, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred_rounded, average='macro', zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred, average='macro')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc_score': roc_auc
        }
        split_metrics.append(metrics)
        roc_auc_scores.append(roc_auc)

        logger.info(f'Metrics for Run {run_name}: {metrics}')

    # Calculate average metrics and confidence intervals
    avg_metrics = {key: np.mean([m[key] for m in split_metrics]) for key in split_metrics[0].keys()}
    roc_auc_mean, roc_auc_lower, roc_auc_upper = compute_2sigma_confidence_interval(roc_auc_scores)

    logger.info(f'Average Metrics: {avg_metrics}')
    logger.info(f'ROC AUC 2-Sigma Confidence Interval: [{roc_auc_lower:.4f}, {roc_auc_upper:.4f}]')

    # Log metrics to MLflow
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_param('dataset_type', dataset_type)
        mlflow.log_param('run_type', RUN_TYPE)
        for key, value in avg_metrics.items():
            mlflow.log_metric(f'avg_{key}', value)
        mlflow.log_metric('roc_auc_2sigma_mean', roc_auc_mean)
        mlflow.log_metric('roc_auc_2sigma_lower', roc_auc_lower)
        mlflow.log_metric('roc_auc_2sigma_upper', roc_auc_upper)


if __name__ == '__main__':
    app()
