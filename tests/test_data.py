import pandas as pd

from flight_delay_prediction.data import (
    FEATURE_COLUMNS,
    SOURCE_COLUMNS,
    TARGET_COLUMN,
    prepare_splits,
)
from flight_delay_prediction.features import (
    fit_category_levels,
    transform_features,
)


def make_flight(
    flight_date: str,
    airline: str = "AA",
    cancelled: int = 0,
    target: int | None = 0,
) -> dict:
    """Create a sample flight record for testing."""
    return {
        "FlightDate": flight_date,
        "DayOfWeek": 1,
        "Reporting_Airline": airline,
        "OriginAirportID": 12478,
        "DestAirportID": 12892,
        "CRSDepTime": 900,
        "CRSArrTime": 1100,
        "CRSElapsedTime": 120,
        "Distance": 800,
        "DepDel15": target,
        "Cancelled": cancelled,
    }


def test_prepare_splits_filters_and_splits_data(tmp_path):
    source_path = tmp_path / "source.csv"
    output_dir = tmp_path / "processed"

    records = [
        make_flight("2024-06-10"),
        make_flight("2025-02-10", airline="DL"),
        make_flight("2025-05-10", airline="UA"),
        make_flight("2024-07-10", cancelled=1),
        make_flight("2024-08-10", target=None),
    ]

    pd.DataFrame(records, columns=SOURCE_COLUMNS).to_csv(
        source_path,
        index=False,
    )

    counts = prepare_splits(
        source_path=source_path,
        output_dir=output_dir,
        chunksize=2,
    )

    assert counts == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }

    train = pd.read_csv(output_dir / "train.csv")
    validation = pd.read_csv(output_dir / "validation.csv")
    test = pd.read_csv(output_dir / "test.csv")

    assert len(train) == 1
    assert len(validation) == 1
    assert len(test) == 1

    assert train.iloc[0]["Reporting_Airline"] == "AA"
    assert validation.iloc[0]["Reporting_Airline"] == "DL"
    assert test.iloc[0]["Reporting_Airline"] == "UA"

    assert list(train.columns) == [
        "FlightDate",
        *FEATURE_COLUMNS,
        TARGET_COLUMN,
    ]


def test_transform_features_handles_unseen_categories():
    train = pd.DataFrame(
        [
            make_flight("2024-06-10", airline="AA"),
            make_flight("2024-07-10", airline="DL"),
        ]
    )

    validation = pd.DataFrame([make_flight("2025-02-10", airline="NEW_AIRLINE")])

    category_levels = fit_category_levels(train)

    train_features = transform_features(train, category_levels)
    validation_features = transform_features(
        validation,
        category_levels,
    )

    assert "__UNKNOWN__" in category_levels["Reporting_Airline"]

    assert validation_features.iloc[0]["Reporting_Airline"] == "__UNKNOWN__"

    assert str(train_features["Reporting_Airline"].dtype) == "category"
    assert str(validation_features["Reporting_Airline"].dtype) == "category"

    assert list(train_features.columns) == FEATURE_COLUMNS
    assert list(validation_features.columns) == FEATURE_COLUMNS
