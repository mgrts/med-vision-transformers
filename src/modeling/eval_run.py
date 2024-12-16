import mlflow
import numpy as np
import typer
from loguru import logger

from src.config import TRACKING_URI

app = typer.Typer(pretty_exceptions_show_locals=False)


def compute_2sigma_confidence_interval(data):
    """
    Compute the confidence interval for a given data array using 2-sigma (two standard deviations).
    """
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)
    margin_of_error = 2 * std_dev  # 2-sigma is approximately 95% confidence interval

    return mean, mean - margin_of_error, mean + margin_of_error


@app.command()
def calculate_confidence_interval(
        metric: str = 'val_loss',
        tracking_uri: str = TRACKING_URI,
        parent_run_id: str = '3f6408c6505f4401a561f8365f5b2bce'
):
    """
    Calculate 2-sigma confidence interval for a specified metric across all nested runs under a parent MLflow run.

    Args:
        metric (str): The metric to calculate the confidence interval for (e.g., 'roc_auc_score', 'val_loss').
        tracking_uri (str): The MLflow tracking URI.
        parent_run_id (str): The parent run ID to fetch nested runs.
    """
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    # Retrieve the parent run to get the experiment ID
    parent_run = client.get_run(parent_run_id)
    experiment_id = parent_run.info.experiment_id

    # Retrieve all child runs
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f'tags.mlflow.parentRunId = "{parent_run_id}"',
    )

    # Extract the specified metric from all nested runs
    metric_values = []
    for run in child_runs:
        if metric in run.data.metrics:
            metric_values.append(run.data.metrics[metric])

    if not metric_values:
        logger.error(f'No values found for metric "{metric}" in the nested runs.')
        return

    metric_values = np.array(metric_values)

    # Compute the 2-sigma confidence interval
    mean, lower_bound, upper_bound = compute_2sigma_confidence_interval(metric_values)

    # Print the results
    logger.info(f'Parent Run ID: {parent_run_id}')
    logger.info(f'Metric: {metric}')
    logger.info(f'Metric Values: {metric_values}')
    logger.info(f'Mean {metric}: {mean:.4f}')
    logger.info(f'2-Sigma Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]')


if __name__ == '__main__':
    app()
