import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import xgboost as xgb

from flight_delay_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN
from flight_delay_prediction.features import (
    fit_category_levels,
    transform_features,
)


def load_training_frame(path: str | Path) -> pd.DataFrame:
    """Load model features and target from a prepared dataset."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    frame = pd.read_csv(path, usecols=required_columns)

    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")

    if frame[TARGET_COLUMN].isna().any():
        raise ValueError(f"Missing target values found in {path}")

    if not frame[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"Invalid target values found in {path}")

    return frame


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"

    return result.stdout.strip()


def train_model(config: dict) -> dict:
    """Train XGBoost, track the experiment, and save model artifacts."""
    training_config = config["training"]
    model_config = config["model"]

    parameters = dict(model_config["parameters"])
    training_settings = model_config["training"]

    model_dir = Path(training_config["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    train_frame = load_training_frame(training_config["train_path"])
    validation_frame = load_training_frame(training_config["validation_path"])

    category_levels = fit_category_levels(train_frame)

    train_features = transform_features(train_frame, category_levels)
    train_target = train_frame[TARGET_COLUMN].to_numpy()

    dtrain = xgb.QuantileDMatrix(
        train_features,
        label=train_target,
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )

    del train_features
    del train_frame
    del train_target

    validation_features = transform_features(
        validation_frame,
        category_levels,
    )
    validation_target = validation_frame[TARGET_COLUMN].to_numpy()

    dvalidation = xgb.QuantileDMatrix(
        validation_features,
        label=validation_target,
        ref=dtrain,
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )

    del validation_features
    del validation_frame
    del validation_target

    mlflow.set_tracking_uri(training_config["tracking_uri"])
    mlflow.set_experiment(training_config["experiment_name"])

    with mlflow.start_run(run_name=training_config["run_name"]) as run:
        mlflow.log_params(parameters)
        mlflow.log_params(training_settings)
        mlflow.log_param("feature_count", len(FEATURE_COLUMNS))
        mlflow.log_param("training_rows", dtrain.num_row())
        mlflow.log_param("validation_rows", dvalidation.num_row())
        mlflow.log_param("git_commit", get_git_commit())

        model = xgb.train(
            params=parameters,
            dtrain=dtrain,
            num_boost_round=training_settings["num_boost_round"],
            evals=[
                (dtrain, "train"),
                (dvalidation, "validation"),
            ],
            early_stopping_rounds=training_settings["early_stopping_rounds"],
            verbose_eval=50,
        )

        model_path = model_dir / "flight_delay_model.json"
        category_path = model_dir / "category_levels.json"

        model.save_model(model_path)

        with category_path.open("w", encoding="utf-8") as file:
            json.dump(category_levels, file, indent=2)

        best_iteration = model.best_iteration
        best_score = float(model.best_score)

        mlflow.log_metric("best_validation_aucpr", best_score)
        mlflow.log_metric("best_iteration", best_iteration)

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(category_path))

        return {
            "run_id": run.info.run_id,
            "model_path": str(model_path),
            "category_path": str(category_path),
            "best_iteration": best_iteration,
            "best_validation_aucpr": best_score,
        }
