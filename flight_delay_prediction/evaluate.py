import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from flight_delay_prediction.baselines import (
    historical_airline_baseline,
    majority_class_baseline,
)
from flight_delay_prediction.data import FEATURE_COLUMNS, TARGET_COLUMN
from flight_delay_prediction.features import transform_features


def average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Calculate average precision for binary classification."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    positive_count = np.sum(y_true == 1)

    if positive_count == 0:
        return 0.0

    order = np.argsort(-y_score, kind="stable")
    sorted_labels = y_true[order]
    sorted_scores = y_score[order]

    cumulative_true_positives = np.cumsum(sorted_labels)
    cumulative_false_positives = (
        np.arange(1, len(sorted_labels) + 1) - cumulative_true_positives
    )

    group_ends = np.r_[
        np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
        len(sorted_scores) - 1,
    ]

    true_positives = cumulative_true_positives[group_ends]
    false_positives = cumulative_false_positives[group_ends]

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positive_count
    previous_recall = np.r_[0.0, recall[:-1]]

    return float(np.sum((recall - previous_recall) * precision))


def roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    """Calculate ROC-AUC using average ranks for tied scores."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    positive_count = np.sum(y_true == 1)
    negative_count = np.sum(y_true == 0)

    if positive_count == 0 or negative_count == 0:
        return 0.0

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    positive_rank_sum = np.sum(ranks[y_true == 1])

    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate binary classification metrics."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_probability = np.asarray(y_probability, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Evaluation data cannot be empty")

    if len(y_true) != len(y_probability):
        raise ValueError("Target and prediction lengths do not match")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Target must contain only binary labels")

    if not np.isfinite(y_probability).all():
        raise ValueError("Predictions contain non-finite values")

    if ((y_probability < 0) | (y_probability > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    predictions = (y_probability >= threshold).astype(np.int8)

    true_positives = int(np.sum((y_true == 1) & (predictions == 1)))
    false_positives = int(np.sum((y_true == 0) & (predictions == 1)))
    false_negatives = int(np.sum((y_true == 1) & (predictions == 0)))
    true_negatives = int(np.sum((y_true == 0) & (predictions == 0)))

    precision = (
        true_positives / (true_positives + false_positives)
        if true_positives + false_positives
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if true_positives + false_negatives
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "pr_auc": average_precision(y_true, y_probability),
        "roc_auc": roc_auc(y_true, y_probability),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def select_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
) -> float:
    """Select the probability threshold that maximizes F1."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_probability = np.asarray(y_probability, dtype=np.float64)

    if len(y_true) == 0 or len(y_true) != len(y_probability):
        raise ValueError("Invalid validation targets or predictions")

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Validation target must contain only binary labels")

    if not np.isfinite(y_probability).all():
        raise ValueError("Validation predictions contain non-finite values")

    if ((y_probability < 0) | (y_probability > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1")

    order = np.argsort(-y_probability, kind="stable")
    scores = y_probability[order]
    labels = y_true[order]

    group_ends = np.r_[
        np.flatnonzero(scores[:-1] != scores[1:]),
        len(scores) - 1,
    ]

    true_positives = np.cumsum(labels)[group_ends]
    false_positives = group_ends + 1 - true_positives
    false_negatives = np.sum(labels) - true_positives

    f1 = (
        2
        * true_positives
        / np.maximum(
            2 * true_positives + false_positives + false_negatives,
            1,
        )
    )

    return float(scores[group_ends[np.argmax(f1)]])


def save_evaluation_plots(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    model: xgb.Booster,
    output_dir: str | Path,
    threshold: float = 0.5,
) -> list[Path]:
    """Save precision-recall, confusion matrix, and feature importance plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = []

    figure, axis = plt.subplots(figsize=(8, 6))

    for name, probabilities in predictions.items():
        order = np.argsort(-probabilities, kind="stable")
        sorted_labels = y_true[order]
        sorted_scores = probabilities[order]

        group_ends = np.r_[
            np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]),
            len(sorted_scores) - 1,
        ]

        true_positives = np.cumsum(sorted_labels)[group_ends]
        false_positives = (
            np.arange(1, len(sorted_labels) + 1)[group_ends] - true_positives
        )

        precision = true_positives / (true_positives + false_positives)
        positive_count = np.sum(y_true == 1)

        recall = (
            true_positives / positive_count
            if positive_count
            else np.zeros_like(true_positives, dtype=np.float64)
        )

        axis.plot(recall, precision, label=name)

    axis.set(
        xlabel="Recall",
        ylabel="Precision",
        title="Precision-Recall Curves",
    )
    axis.legend()
    axis.grid(alpha=0.3)

    path = output_dir / "precision_recall_curve.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(path)

    probabilities = predictions["xgboost"]
    predicted_labels = (probabilities >= threshold).astype(np.int8)

    matrix = np.array(
        [
            [
                np.sum((y_true == 0) & (predicted_labels == 0)),
                np.sum((y_true == 0) & (predicted_labels == 1)),
            ],
            [
                np.sum((y_true == 1) & (predicted_labels == 0)),
                np.sum((y_true == 1) & (predicted_labels == 1)),
            ],
        ]
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)

    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
            )

    axis.set_xticks([0, 1], labels=["On time", "Delayed"])
    axis.set_yticks([0, 1], labels=["On time", "Delayed"])
    axis.set(
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"XGBoost Confusion Matrix (threshold={threshold:.4f})",
    )

    path = output_dir / "confusion_matrix.png"
    figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    plot_paths.append(path)

    importance = model.get_score(importance_type="gain")

    if importance:
        importance_series = pd.Series(importance).sort_values()

        figure, axis = plt.subplots(figsize=(9, 6))
        importance_series.plot(kind="barh", ax=axis)
        axis.set(
            xlabel="Average gain",
            title="XGBoost Feature Importance",
        )

        path = output_dir / "feature_importance.png"
        figure.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        plot_paths.append(path)

    return plot_paths


def get_git_commit() -> str:
    """Return the current Git commit hash."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def evaluate_model(
    test_path: str | Path,
    train_path: str | Path,
    validation_path: str | Path,
    model_path: str | Path,
    category_path: str | Path,
    output_dir: str | Path = "plots",
    tracking_uri: str = "http://127.0.0.1:8080",
    experiment_name: str = "flight-delay-prediction",
) -> dict:
    """Select thresholds on validation data and evaluate on test data."""
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]

    validation_frame = pd.read_csv(
        validation_path,
        usecols=required_columns,
    )
    test_frame = pd.read_csv(
        test_path,
        usecols=required_columns,
    )
    train_frame = pd.read_csv(
        train_path,
        usecols=["Reporting_Airline", TARGET_COLUMN],
    )

    if validation_frame.empty or test_frame.empty or train_frame.empty:
        raise ValueError("Training, validation, and test datasets must not be empty")

    for name, frame in [
        ("Validation", validation_frame),
        ("Test", test_frame),
    ]:
        if not frame[TARGET_COLUMN].isin([0, 1]).all():
            raise ValueError(f"{name} data contains invalid target values")

    validation_rows = len(validation_frame)

    with Path(category_path).open(encoding="utf-8") as file:
        category_levels = json.load(file)

    model = xgb.Booster()
    model.load_model(model_path)

    # Select thresholds using validation data.
    validation_features = transform_features(
        validation_frame,
        category_levels,
    )
    dvalidation = xgb.DMatrix(
        validation_features,
        enable_categorical=True,
    )

    y_validation = validation_frame[TARGET_COLUMN].to_numpy(dtype=np.int8)

    validation_predictions = {
        "xgboost": model.predict(dvalidation),
        "majority_baseline": majority_class_baseline(
            train_frame[TARGET_COLUMN],
            validation_frame,
        ).to_numpy(),
        "historical_airline_baseline": historical_airline_baseline(
            train_frame,
            validation_frame,
        ).to_numpy(),
    }

    thresholds = {
        name: select_threshold(y_validation, probabilities)
        for name, probabilities in validation_predictions.items()
    }

    # Save the selected XGBoost threshold for prediction.
    threshold_path = Path(model_path).parent / "threshold.json"
    threshold_path.write_text(
        json.dumps({"xgboost": thresholds["xgboost"]}, indent=2),
        encoding="utf-8",
    )

    del validation_frame, validation_features, dvalidation
    del validation_predictions, y_validation

    # Evaluate on the untouched test data.
    y_true = test_frame[TARGET_COLUMN].to_numpy(dtype=np.int8)

    test_features = transform_features(test_frame, category_levels)
    dtest = xgb.DMatrix(
        test_features,
        enable_categorical=True,
    )

    predictions = {
        "xgboost": model.predict(dtest),
        "majority_baseline": majority_class_baseline(
            train_frame[TARGET_COLUMN],
            test_frame,
        ).to_numpy(),
        "historical_airline_baseline": historical_airline_baseline(
            train_frame,
            test_frame,
        ).to_numpy(),
    }

    results = {
        name: calculate_metrics(
            y_true,
            probabilities,
            threshold=thresholds[name],
        )
        for name, probabilities in predictions.items()
    }

    plot_paths = save_evaluation_plots(
        y_true,
        predictions,
        model,
        output_dir,
        threshold=thresholds["xgboost"],
    )

    # Log evaluation results separately from the training run.
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="test-set-evaluation") as run:
        mlflow.set_tags(
            {
                "run_type": "evaluation",
                "evaluation_dataset": "test",
                "git_commit": get_git_commit(),
            }
        )

        mlflow.log_params(
            {
                **{f"{name}_threshold": value for name, value in thresholds.items()},
                "test_rows": len(test_frame),
                "validation_rows_for_threshold_selection": validation_rows,
                "train_rows_for_baselines": len(train_frame),
                "model_type": "XGBoost",
                "threshold_selection_metric": "F1",
            }
        )

        for model_name, metrics in results.items():
            mlflow.log_metrics(
                {
                    f"{model_name}.{metric_name}": float(value)
                    for metric_name, value in metrics.items()
                }
            )

        for plot_path in plot_paths:
            mlflow.log_artifact(
                str(plot_path),
                artifact_path="evaluation_plots",
            )

        mlflow.log_artifact(str(threshold_path), artifact_path="model")

        evaluation_run_id = run.info.run_id

    return {
        "metrics": results,
        "thresholds": thresholds,
        "plots": [str(path) for path in plot_paths],
        "mlflow_run_id": evaluation_run_id,
    }
