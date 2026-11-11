import pandas as pd

from flight_delay_prediction.data import TARGET_COLUMN


def majority_class_baseline(
    train_target: pd.Series,
    evaluation: pd.DataFrame,
) -> pd.Series:
    """Return a constant probability based on the training class balance."""
    if train_target.empty:
        raise ValueError("Training target cannot be empty")

    positive_rate = float(train_target.mean())

    return pd.Series(
        positive_rate,
        index=evaluation.index,
        dtype="float64",
    )


def historical_airline_baseline(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> pd.Series:
    """Predict delay probability using training-set airline delay rates."""
    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"Training data must contain {TARGET_COLUMN}")

    if "Reporting_Airline" not in evaluation.columns:
        raise ValueError("Evaluation data must contain Reporting_Airline")

    overall_delay_rate = float(train[TARGET_COLUMN].mean())

    airline_delay_rates = train.groupby(
        "Reporting_Airline",
        observed=True,
    )[TARGET_COLUMN].mean()

    predictions = evaluation["Reporting_Airline"].map(airline_delay_rates)

    return predictions.fillna(overall_delay_rate).astype("float64")
