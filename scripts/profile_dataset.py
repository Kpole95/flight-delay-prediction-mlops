from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "interim" / "flight_delays_2024_2025.csv"

CHUNK_SIZE = 200_000

FEATURE_COLUMNS = [
    "DayOfWeek",
    "Reporting_Airline",
    "OriginAirportID",
    "DestAirportID",
    "CRSDepTime",
    "CRSArrTime",
    "CRSElapsedTime",
    "Distance",
]

REQUIRED_COLUMNS = [
    "FlightDate",
    "Cancelled",
    "DepDel15",
    *FEATURE_COLUMNS,
]


def profile_dataset() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(DATA_FILE)

    total_rows = 0
    cancelled_counts = Counter()
    target_counts = Counter()
    eligible_target_counts = Counter()
    monthly_counts = Counter()
    missing_features = Counter()

    eligible_rows = 0
    invalid_target_rows = 0
    first_date = None
    last_date = None

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            DATA_FILE,
            usecols=REQUIRED_COLUMNS,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):
        total_rows += len(chunk)

        dates = pd.to_datetime(
            chunk["FlightDate"],
            format="%Y-%m-%d",
            errors="coerce",
        )

        first_date = dates.min() if first_date is None else min(first_date, dates.min())
        last_date = dates.max() if last_date is None else max(last_date, dates.max())

        monthly_counts.update(
            dates.dt.to_period("M").astype(str).value_counts().to_dict()
        )

        cancelled = pd.to_numeric(chunk["Cancelled"], errors="coerce")
        target = pd.to_numeric(chunk["DepDel15"], errors="coerce")

        cancelled_counts.update(
            cancelled.astype("string").fillna("<MISSING>").value_counts().to_dict()
        )

        target_counts.update(
            target.astype("string").fillna("<MISSING>").value_counts().to_dict()
        )

        valid_target = target.isin([0, 1])
        eligible = cancelled.eq(0) & valid_target

        eligible_rows += int(eligible.sum())
        invalid_target_rows += int((~valid_target).sum())

        eligible_target_counts.update(
            target.loc[eligible].astype(int).value_counts().to_dict()
        )

        for column in FEATURE_COLUMNS:
            missing_features[column] += int(chunk.loc[eligible, column].isna().sum())

        print(f"Processed {total_rows:,} records (chunk {chunk_number}).")

    print("\n========== DATASET PROFILE ==========")
    print(f"Total records: {total_rows:,}")
    print(f"Date range: {first_date.date()} to {last_date.date()}")
    print(f"Eligible records: {eligible_rows:,}")
    print(f"Invalid or missing targets: {invalid_target_rows:,}")

    print("\n========== CANCELLATIONS ==========")
    for value, count in sorted(cancelled_counts.items()):
        print(f"{value}: {count:,}")

    print("\n========== ALL TARGET VALUES ==========")
    for value, count in sorted(target_counts.items()):
        print(f"{value}: {count:,}")

    print("\n========== ELIGIBLE TARGET DISTRIBUTION ==========")
    for value, count in sorted(eligible_target_counts.items()):
        percentage = count / eligible_rows * 100
        print(f"{value}: {count:,} ({percentage:.2f}%)")

    print("\n========== MISSING FEATURE VALUES ==========")
    for column, count in missing_features.items():
        print(f"{column}: {count:,}")

    print("\n========== MONTHLY RECORD COUNTS ==========")
    for month, count in sorted(monthly_counts.items()):
        print(f"{month}: {count:,}")


if __name__ == "__main__":
    profile_dataset()
