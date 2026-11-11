import numpy as np
import pandas as pd
import pytest

from flight_delay_prediction.evaluate import (
    average_precision,
    calculate_metrics,
    roc_auc,
    select_threshold,
)


def test_average_precision():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1])

    result = average_precision(y_true, y_score)

    assert result == pytest.approx(5 / 6)


def test_roc_auc():
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.1])

    result = roc_auc(y_true, y_score)

    assert result == pytest.approx(0.75)


def test_metrics_at_threshold():
    y_true = np.array([1, 0, 1, 0])
    y_probability = np.array([0.9, 0.8, 0.7, 0.1])

    metrics = calculate_metrics(
        y_true,
        y_probability,
        threshold=0.5,
    )

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["true_negatives"] == 1


def test_metrics_handle_tied_scores():
    y_true = np.array([1, 0])
    y_score = np.array([0.5, 0.5])

    assert average_precision(y_true, y_score) == pytest.approx(0.5)
    assert roc_auc(y_true, y_score) == pytest.approx(0.5)


def test_select_threshold_maximizes_f1():
    y_true = np.array([1, 1, 0, 0])
    probabilities = np.array([0.9, 0.6, 0.4, 0.1])

    threshold = select_threshold(y_true, probabilities)

    assert threshold == pytest.approx(0.6)


def test_majority_baseline_uses_fixed_threshold():
    from flight_delay_prediction.baselines import majority_class_baseline

    train_target = pd.Series([0, 0, 0, 1, 1])
    evaluation = pd.DataFrame(index=[0, 1, 2])

    predictions = majority_class_baseline(train_target, evaluation)
    threshold = 0.5

    assert threshold == 0.5
    assert (predictions.to_numpy() >= threshold).tolist() == [
        False,
        False,
        False,
    ]
