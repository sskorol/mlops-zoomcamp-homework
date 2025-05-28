import os
import pickle
import click
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(params, X_train, y_train, X_val, y_val, X_test, y_test):
    with mlflow.start_run():
        mlflow.log_params(params)

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)
        test_pred = rf.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        test_rmse = np.sqrt(test_mse)

        mlflow.log_metric("train_rmse", train_rmse)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("rmse", test_rmse)

        mlflow.sklearn.log_model(rf, "model")

        return {
            'model': rf,
            'test_rmse': test_rmse,
            'run_id': mlflow.active_run().info.run_id
        }


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--num_candidates",
    default=5,
    help="Number of top models from hyperopt to evaluate"
)
def run_model_registration(data_path: str, num_candidates: int):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    client = MlflowClient()

    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(f"Could not find experiment '{HPO_EXPERIMENT_NAME}'. Please run hpo.py first.")

    best_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=num_candidates
    )

    if len(best_runs) == 0:
        raise ValueError(f"No runs found in {HPO_EXPERIMENT_NAME} experiment.")

    print(f"Found {len(best_runs)} top hyperparameter combinations to evaluate")

    final_model_results = []

    for i, run in enumerate(best_runs, 1):
        validation_rmse = run.data.metrics.get("rmse", "N/A")
        print(f"Evaluating candidate {i}/{len(best_runs)} - Original validation RMSE: {validation_rmse}")

        try:
            params = {}
            for param_name, param_value in run.data.params.items():
                if param_name in RF_PARAMS:
                    if param_name in ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf']:
                        params[param_name] = int(float(param_value))
                    elif param_name == 'random_state':
                        params[param_name] = int(float(param_value))

            result = train_and_log_model(params, X_train, y_train, X_val, y_val, X_test, y_test)
            result['original_validation_rmse'] = validation_rmse
            final_model_results.append(result)

            print(f"  Test RMSE: {result['test_rmse']:.4f}")

        except Exception as e:
            print(f"  Error training model: {str(e)}")
            continue

    if not final_model_results:
        raise ValueError("No models could be successfully trained and evaluated.")

    best_model = min(final_model_results, key=lambda x: x['test_rmse'])

    print(f"\nBest model test RMSE: {best_model['test_rmse']:.4f}")

    model_name = "taxi-trip-duration-predictor"
    model_uri = f"runs:/{best_model['run_id']}/model"

    try:
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )

        print(f"Model registered successfully!")
        print(f"  Name: {model_name}")
        print(f"  Version: {model_version.version}")
        print(f"  Test RMSE: {best_model['test_rmse']:.4f}")

        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=f"Random Forest taxi duration predictor. Test RMSE: {best_model['test_rmse']:.4f} minutes."
        )

    except Exception as e:
        print(f"Error registering model: {str(e)}")


if __name__ == '__main__':
    run_model_registration()
