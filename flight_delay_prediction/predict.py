import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from flight_delay_prediction.features import transform_features


def predict_flights(
    flights: pd.DataFrame,
    model_path: str | Path = "models/flight_delay_model.json",
    category_path: str | Path = "models/category_levels.json",
    threshold_path: str | Path = "models/threshold.json",
    threshold: float | None = None,
) -> pd.DataFrame:
    """Predict departure delay probabilities for flight records."""
    with Path(category_path).open(encoding="utf-8") as file:
        category_levels = json.load(file)

    if threshold is None:
        with Path(threshold_path).open(encoding="utf-8") as file:
            threshold = json.load(file)["xgboost"]

    model = xgb.Booster()
    model.load_model(str(model_path))

    features = transform_features(flights, category_levels)
    probabilities = model.predict(xgb.DMatrix(features, enable_categorical=True))

    return pd.DataFrame(
        {
            "delay_probability": probabilities,
            "predicted_delay": (probabilities >= threshold).astype(int),
        }
    )
