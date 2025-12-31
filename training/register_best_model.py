import mlflow
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "fx_forecasting"
REGISTERED_MODEL_NAME = "fx_exchange_rate_forecaster"


def main():
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.rmse_mean ASC"],
        max_results=1
    )

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_rmse = best_run.data.metrics["rmse_mean"]

    print(f"Best run ID: {best_run_id}")
    print(f"Best RMSE: {best_rmse:.4f}")

    model_uri = f"runs:/{best_run_id}/model"

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME
    )

    print(f"Registered model version: {result.version}")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print("Model promoted to PRODUCTION")


if __name__ == "__main__":
    main()
