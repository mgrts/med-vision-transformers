import mlflow
import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.config import (BASE_MODEL_NAME, BATCH_SIZE, IMAGE_SIZE, MODELS_DIR,
                        SEGMENTED_TEST_ANNOTATIONS_PATH,
                        SEGMENTED_TEST_DATA_DIR, TRACKING_URI)
from src.modeling.data_processing import ImageDatasetCOCO, collate_fn
from src.modeling.models import (MultiLabelClassificationTransformer,
                                 MultiTaskTransformer)
from src.modeling.train import collect_predictions, compute_binary_metrics
from src.modeling.utils import DEVICE, EVAL_TRANSFORM, confidence_interval

app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
        parent_run_id: str = typer.Option(..., help='MLflow run id of the training run to evaluate'),
        training_task: str = 'multi-task',
        dataset_type: str = 'coco',
        batch_size: int = BATCH_SIZE,
):
    """Evaluate each cross-validation fold of a training run on the HELD-OUT test set.

    Reports per-fold binary metrics (PR-AUC, ROC-AUC, F1, etc.) and their t-based confidence
    interval. The training run id is passed explicitly (no machine-specific hashes are baked
    into the source); evaluation never touches the train/validation data.
    """
    if training_task not in ('classification', 'multi-task'):
        raise typer.BadParameter("training_task must be 'classification' or 'multi-task'")
    if dataset_type != 'coco':
        raise typer.BadParameter(f'Unsupported dataset type: {dataset_type}')

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment('evaluate_clf')
    experiment = mlflow.get_experiment_by_name('evaluate_clf')
    client = mlflow.tracking.MlflowClient()

    try:
        parent_run = client.get_run(parent_run_id)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable message
        logger.error(
            f'Could not load run {parent_run_id}: {exc}. Pass a valid --parent-run-id from a '
            f'completed training run (its weights must exist under MODELS_DIR).'
        )
        raise typer.Exit(code=1)

    child_runs = client.search_runs(
        experiment_ids=[parent_run.info.experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
    )
    if not child_runs:
        logger.error(f'No fold runs found under parent run {parent_run_id}')
        raise typer.Exit(code=1)

    # Held-out official test split only — no train/validation data is evaluated.
    test_dataset = ImageDatasetCOCO(
        annotation_file=SEGMENTED_TEST_ANNOTATIONS_PATH,
        image_dir=SEGMENTED_TEST_DATA_DIR,
        transform=EVAL_TRANSFORM,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    num_classes = test_dataset.num_classes  # 1 (binary)

    per_fold = []
    for run in child_runs:
        run_name = run.data.tags.get('mlflow.runName', run.info.run_id)
        model_files = [a.path for a in client.list_artifacts(run.info.run_id) if a.path.endswith('.pth')]
        if not model_files:
            logger.warning(f'Run {run_name} has no model artifact; skipping')
            continue

        model_uri = MODELS_DIR / parent_run_id / model_files[0]
        if not model_uri.exists():
            logger.error(
                f'Model artifact not found at {model_uri}. The run weights are not available '
                f'locally; retrain or sync MODELS_DIR before evaluating.'
            )
            raise typer.Exit(code=1)

        base_model = AutoModel.from_pretrained(
            BASE_MODEL_NAME, add_pooling_layer=False, attn_implementation='eager'
        )
        if training_task == 'classification':
            model = MultiLabelClassificationTransformer(base_model, num_classes=num_classes)
        else:
            model = MultiTaskTransformer(base_model, image_size=IMAGE_SIZE, num_classes=num_classes)
        model.load_state_dict(torch.load(model_uri, map_location=DEVICE))
        model.to(DEVICE)

        y_prob, y_true = collect_predictions(model, test_loader, training_task)
        metrics, _ = compute_binary_metrics(y_true, y_prob)
        per_fold.append(metrics)
        logger.info(f'Test metrics for {run_name}: {metrics}')

    if not per_fold:
        logger.error('No fold models could be evaluated.')
        raise typer.Exit(code=1)

    with mlflow.start_run(experiment_id=experiment.experiment_id):
        mlflow.log_param('dataset_type', dataset_type)
        mlflow.log_param('training_task', training_task)
        mlflow.log_param('parent_run_id', parent_run_id)
        for key in per_fold[0]:
            values = [m[key] for m in per_fold if m[key] == m[key]]  # drop NaNs
            if not values:
                continue
            mean, lower, upper = confidence_interval(values)
            mlflow.log_metric(f'test_avg_{key}', round(mean, 4))
            mlflow.log_metric(f'test_{key}_lower', round(lower, 4))
            mlflow.log_metric(f'test_{key}_upper', round(upper, 4))
            logger.info(f'{key}: mean={mean:.4f}  95% CI=[{lower:.4f}, {upper:.4f}]')


if __name__ == '__main__':
    app()
