import pandas as pd

from flight_delay_prediction.data import FEATURE_COLUMNS

CATEGORICAL_COLUMNS = [
    "Reporting_Airline",
    "OriginAirportID",
    "DestAirportID",
]

UNKNOWN_CATEGORY = "__UNKNOWN__"


def fit_category_levels(
    train: pd.DataFrame,
) -> dict[str, list[str]]:
    """Learn categorical levels from training data."""
    category_levels = {}

    for column in CATEGORICAL_COLUMNS:
        levels = sorted(train[column].dropna().astype(str).unique().tolist())
        levels.append(UNKNOWN_CATEGORY)
        category_levels[column] = levels

    return category_levels


def transform_features(
    frame: pd.DataFrame,
    category_levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Prepare model features using previously learned category levels."""
    missing_columns = set(FEATURE_COLUMNS) - set(frame.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing features: {sorted(missing_columns)}")

    features = frame[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_COLUMNS:
        values = features[column].astype("string")
        values = values.where(
            values.isin(category_levels[column]),
            UNKNOWN_CATEGORY,
        )

        features[column] = pd.Categorical(
            values,
            categories=category_levels[column],
        )

    numeric_columns = [
        column for column in FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
    ]

    for column in numeric_columns:
        features[column] = pd.to_numeric(
            features[column],
            errors="raise",
        )

    return features
