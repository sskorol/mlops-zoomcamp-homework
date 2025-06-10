"""
Utility functions to inspect MLflow models and answer homework questions
"""

import mlflow
import os
from mlflow import MlflowClient
from prefect import task, flow
from prefect.logging import get_run_logger


@task
def inspect_latest_model(experiment_name: str = "homework3-yellow-taxi-prefect"):
    """Inspect the latest model for homework answers"""
    logger = get_run_logger()

    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = MlflowClient()

    try:
        # Get experiment
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            logger.error(f"Experiment {experiment_name} not found")
            return None

        # Get latest run
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )

        if not runs:
            logger.error("No runs found")
            return None

        latest_run = runs[0]
        run_id = latest_run.info.run_id

        logger.info(f"Latest run ID: {run_id}")

        # Get metrics and parameters
        metrics = latest_run.data.metrics
        params = latest_run.data.params

        print(f"\n=== HOMEWORK ANSWERS ===")
        print(f"Question 5 - Model intercept: {metrics.get('intercept', 'Not found')}")
        print(f"RMSE: {metrics.get('rmse', 'Not found')}")

        # Try to get model size from artifacts
        try:
            artifacts = client.list_artifacts(run_id, path="model")
            for artifact in artifacts:
                if artifact.path == "model/MLmodel":
                    # Download MLmodel file to check size
                    artifact_path = client.download_artifacts(run_id, "model/MLmodel")
                    model_size = os.path.getsize(artifact_path)
                    print(f"Question 6 - Model size: {model_size} bytes")
                    break
        except Exception as e:
            logger.warning(f"Could not get model size: {e}")
            print("Question 6 - Check MLflow UI for model size")

        print(f"\nCheck MLflow UI at http://localhost:5000 for full details")
        print(f"Run ID: {run_id}")

        return {
            "run_id": run_id,
            "metrics": metrics,
            "params": params
        }

    except Exception as e:
        logger.error(f"Error inspecting model: {e}")
        return None


@flow(name="Model Inspection")
def model_inspection_flow(experiment_name: str = "homework3-yellow-taxi-prefect"):
    """Flow to inspect models and provide homework answers"""
    logger = get_run_logger()
    logger.info("Starting model inspection for homework answers")

    results = inspect_latest_model(experiment_name)

    if results:
        logger.info("Model inspection completed successfully")
    else:
        logger.error("Model inspection failed")

    return results


if __name__ == "__main__":
    # For standalone execution
    model_inspection_flow()
