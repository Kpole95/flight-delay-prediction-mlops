import pandas as pd
import pytest

from flight_delay_prediction.baselines import (
    historical_airline_baseline,
    majority_class_baseline,
)


def test_majority_class_baseline_predicts_majority_class():
    train_target = pd.Series([0, 0, 0, 1, 1])
    evaluation = pd.DataFrame(index=[10, 20, 30])

    predictions = majority_class_baseline(train_target, evaluation)

    assert predictions.tolist() == [0.0, 0.0, 0.0]
    assert predictions.index.tolist() == [10, 20, 30]


def test_historical_airline_baseline_uses_training_rates():
    train = pd.DataFrame(
        {
            "Reporting_Airline": ["AA", "AA", "AA", "DL", "DL"],
            "DepDel15": [1, 0, 1, 0, 0],
        }
    )

    evaluation = pd.DataFrame(
        {
            "Reporting_Airline": ["AA", "DL", "UA"],
        }
    )

    predictions = historical_airline_baseline(train, evaluation)

    assert predictions.tolist() == pytest.approx([2 / 3, 0.0, 0.4])


def test_majority_baseline_rejects_empty_training_data():
    train_target = pd.Series(dtype="int8")
    evaluation = pd.DataFrame(index=[0])

    with pytest.raises(ValueError, match="cannot be empty"):
        majority_class_baseline(train_target, evaluation)
