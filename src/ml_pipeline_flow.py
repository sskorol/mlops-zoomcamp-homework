import os
import pickle
from typing import Any, Dict

import pandas as pd
import numpy as np
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlflow import MlflowClient

from prefect import task, flow
from prefect.logging import get_run_logger

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "homework3-yellow-taxi-prefect"


def dump_pickle(obj: Any, filename: str | Path) -> None:
    """
    Save an object as a pickle file using pathlib's write_bytes method

    Args:
        obj: The object to save
        filename: Path to save the file (can be string or Path object)
    """
    # Convert to Path object to use modern path handling
    path = Path(filename)

    # Serialize the object to bytes, then write to file
    serialized_data = pickle.dumps(obj)
    path.write_bytes(serialized_data)


def load_pickle(filename: str | Path) -> Any:
    """
    Load an object from a pickle file using pathlib's read_bytes method

    Args:
        filename: Path to the pickle file (can be string or Path object)

    Returns:
        The deserialized object
    """
    # Convert to Path object and read bytes
    path = Path(filename)
    serialized_data = path.read_bytes()

    # Deserialize the bytes back to an object
    return pickle.loads(serialized_data)


@task
def setup_mlflow():
    """Setup MLflow tracking and return configuration details"""
    logger = get_run_logger()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow experiment: {EXPERIMENT_NAME}")

    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": EXPERIMENT_NAME
    }


@task
def read_dataframe(filepath: str) -> tuple[pd.DataFrame, dict]:
    """
    Read and preprocess data, returning both the dataframe and homework metrics
    This function handles Questions 3 and 4 from the homework
    """
    logger = get_run_logger()

    # Load the data
    df = pd.read_parquet(filepath)
    initial_records = len(df)

    # Answer to Question 3: How many records did we load?
    logger.info(f"Question 3 Answer: Loaded {initial_records:,} records")
    print(f"🎯 QUESTION 3: Loaded {initial_records:,} records")

    # Calculate duration using yellow taxi columns (tpep_*)
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    # Filter outliers (duration between 1 and 60 minutes)
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    # Convert categorical features to strings
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    final_records = len(df)

    # Answer to Question 4: What's the size after data preparation?
    logger.info(f"Question 4 Answer: After preprocessing: {final_records:,} records")
    print(f"🎯 QUESTION 4: After preprocessing: {final_records:,} records")

    # Return both the processed dataframe and the homework metrics
    homework_metrics = {
        "question_3_initial_records": initial_records,
        "question_4_final_records": final_records
    }

    return df, homework_metrics


@task
def preprocess_features(df: pd.DataFrame, fit_dv: bool = True, dv_path: str = None) -> tuple:
    """Create feature matrix using DictVectorizer"""
    logger = get_run_logger()

    # Use pickup and dropoff locations separately (as per homework requirements)
    categorical = ['PULocationID', 'DOLocationID']
    numerical = []  # No numerical features for basic homework

    dicts = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        dv = DictVectorizer()
        X = dv.fit_transform(dicts)
        logger.info("Fitted new DictVectorizer")
    else:
        dv = load_pickle(dv_path)
        X = dv.transform(dicts)
        logger.info("Used existing DictVectorizer")

    y = df['duration'].values

    logger.info(f"Feature matrix shape: {X.shape}")

    return X, y, dv


@task
def train_model_with_mlflow(X, y, dv) -> dict:
    """Train model with detailed MLflow debugging"""
    logger = get_run_logger()

    with mlflow.start_run() as run:
        # Train the model as before
        model = LinearRegression()
        model.fit(X, y)

        # Calculate metrics
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        model_intercept = model.intercept_

        # Log parameters and metrics (this works)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("intercept", model_intercept)

        logger.info("Successfully logged parameters and metrics")

        # Now try to save artifacts with detailed logging
        models_dir = Path("/app/models")
        models_dir.mkdir(exist_ok=True)

        dv_path = models_dir / "dv.pkl"
        model_path = models_dir / "model.pkl"

        # Save files locally first
        dump_pickle(dv, dv_path)
        dump_pickle(model, model_path)
        logger.info(
            f"Saved local files: {dv_path} ({dv_path.stat().st_size} bytes), {model_path} ({model_path.stat().st_size} bytes)"
        )

        # Try to log simple artifact first
        try:
            logger.info("Attempting to log DictVectorizer artifact...")
            mlflow.log_artifact(str(dv_path), artifact_path="preprocessor")
            logger.info("✅ Successfully logged DictVectorizer artifact")
        except Exception as e:
            logger.error(f"❌ Failed to log DictVectorizer artifact: {e}")
            logger.error(f"Exception type: {type(e)}")

        # Try to log the sklearn model
        try:
            logger.info("Attempting to log sklearn model...")
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="yellow-taxi-duration-hw3"
            )
            logger.info("✅ Successfully logged sklearn model")
        except Exception as e:
            logger.error(f"❌ Failed to log sklearn model: {e}")
            logger.error(f"Exception type: {type(e)}")

        return {
            "run_id": run.info.run_id,
            "rmse": rmse,
            "intercept": model_intercept,
        }


@task
def get_model_size_from_mlflow(run_id: str) -> dict:
    """Extract the actual model.pkl file size - the correct model_size_bytes"""
    logger = get_run_logger()
    client = MlflowClient()

    options = {14534: "14,534", 9534: "9,534", 4534: "4,534", 1534: "1,534"}
    closest_option = 0
    out_model_size: Dict[str, Any] = {
        "model_size_bytes": closest_option,
        "closest_homework_option": closest_option,
    }

    try:
        artifacts = client.list_artifacts(run_id, path="model")

        for artifact in artifacts:
            logger.info(f"Found model artifact: {artifact.path} ({artifact.file_size} bytes)")

            if artifact.path.endswith("model.pkl") or "model.pkl" in artifact.path:
                model_size = artifact.file_size

                logger.info(f"Question 6 Answer: Model size: {model_size} bytes")
                print(f"🎯 QUESTION 6: Model size: {model_size} bytes")

                closest_option = min(options.keys(), key=lambda x: abs(x - model_size))

                logger.info(f"Closest homework option: {options[closest_option]}")
                print(f"             Closest option: {options[closest_option]}")

                out_model_size["model_size_bytes"] = model_size
                out_model_size["closest_homework_option"] = options[closest_option]
    except Exception as e:
        logger.error(f"Error getting model size: {e}")

    return out_model_size


@task
def generate_homework_summary(homework_metrics: dict, training_results: dict, model_size_info: dict) -> dict:
    """
    Generate a comprehensive summary of all homework answers
    This creates a complete report for submission
    """
    logger = get_run_logger()

    # Compile all homework answers
    homework_answers = {
        "question_1_orchestrator": "Prefect",
        "question_2_version": "3.4.5",  # From requirements.txt
        "question_3_records_loaded": homework_metrics["question_3_initial_records"],
        "question_4_records_after_prep": homework_metrics["question_4_final_records"],
        "question_5_model_intercept": round(training_results["intercept"], 2),
        "question_6_model_size": model_size_info.get("model_size_bytes"),
        "question_6_closest_option": model_size_info.get("closest_homework_option")
    }

    # Generate a formatted summary
    print("\n" + "=" * 60)
    print("🎓 MLOPS ZOOMCAMP HOMEWORK 3 - COMPLETE ANSWERS")
    print("=" * 60)
    print(f"Question 1 (Orchestrator):     {homework_answers['question_1_orchestrator']}")
    print(f"Question 2 (Version):          {homework_answers['question_2_version']}")
    print(f"Question 3 (Records Loaded):   {homework_answers['question_3_records_loaded']:,}")
    print(f"Question 4 (After Prep):       {homework_answers['question_4_records_after_prep']:,}")
    print(f"Question 5 (Model Intercept):  {homework_answers['question_5_model_intercept']}")
    print(f"Question 6 (Model Size):       {homework_answers['question_6_model_size']} bytes")
    print(f"           (Closest Option):   {homework_answers['question_6_closest_option']}")
    print("=" * 60)
    print("✅ All homework requirements completed successfully!")
    print("=" * 60)

    # Log the summary for permanent record
    logger.info("Homework Summary Generated:")
    for key, value in homework_answers.items():
        logger.info(f"  {key}: {value}")

    return homework_answers


@flow(name="Complete ML Workflow with Homework Answers")
def complete_ml_workflow_with_answers(
        year: int = 2023,
        month: int = 3,
        taxi_type: str = "yellow"
) -> dict:
    """
    Complete ML workflow that captures all homework answers
    This is the main pipeline that demonstrates all requirements
    """
    logger = get_run_logger()
    logger.info(f"Starting complete ML workflow for {taxi_type} taxi {year}-{month:02d}")

    print(f"\n🚀 Starting MLOps Homework 3 Pipeline")
    print(f"   Dataset: {taxi_type.title()} taxi data for {year}-{month:02d}")

    # First ensure we have data
    from data_flow import download_taxi_data
    data_filepath = download_taxi_data(year, month, taxi_type)

    # Setup MLflow
    setup_mlflow()

    # Process data and capture homework metrics
    df, homework_metrics = read_dataframe(data_filepath)
    X, y, dv = preprocess_features(df, fit_dv=True)

    # Train model and capture training results
    training_results = train_model_with_mlflow(X, y, dv)

    # Get model size information for Question 6
    model_size_info = get_model_size_from_mlflow(training_results["run_id"])

    # Generate comprehensive homework summary
    homework_summary = generate_homework_summary(
        homework_metrics,
        training_results,
        model_size_info
    )

    # Add run metadata to summary
    homework_summary.update(
        {
            "mlflow_run_id": training_results["run_id"],
            "mlflow_experiment": EXPERIMENT_NAME,
            "data_filepath": data_filepath,
            "pipeline_status": "completed_successfully"
        }
    )

    logger.info("Complete workflow with homework answers finished successfully")
    return homework_summary


if __name__ == "__main__":
    # For local testing
    results = complete_ml_workflow_with_answers()
    print(f"\n📋 Pipeline completed with all homework answers captured!")
    print(f"🔍 MLflow run ID: {results['mlflow_run_id']}")
