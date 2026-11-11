import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

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

TARGET_COLUMN = "DepDel15"
DATE_COLUMN = "FlightDate"
CANCELLED_COLUMN = "Cancelled"

SOURCE_COLUMNS = [
    DATE_COLUMN,
    *FEATURE_COLUMNS,
    TARGET_COLUMN,
    CANCELLED_COLUMN,
]

TRAIN_END = pd.Timestamp("2025-01-01")
VALIDATION_END = pd.Timestamp("2025-04-01")
TEST_END = pd.Timestamp("2025-07-01")

SPLIT_NAMES = ("train", "validation", "test")


def _get_split_dates(dates: pd.Series) -> dict[str, pd.Series]:
    """Return chronological split masks for a series of flight dates."""
    return {
        "train": dates < TRAIN_END,
        "validation": (dates >= TRAIN_END) & (dates < VALIDATION_END),
        "test": (dates >= VALIDATION_END) & (dates < TEST_END),
    }


def _write_chunk(
    chunk: pd.DataFrame,
    output_paths: dict[str, Path],
    written_splits: set[str],
) -> dict[str, int]:
    """Filter a chunk and append eligible records to their split files."""
    chunk = chunk.copy()

    chunk[DATE_COLUMN] = pd.to_datetime(
        chunk[DATE_COLUMN],
        errors="coerce",
    )
    chunk[TARGET_COLUMN] = pd.to_numeric(
        chunk[TARGET_COLUMN],
        errors="coerce",
    )
    chunk[CANCELLED_COLUMN] = pd.to_numeric(
        chunk[CANCELLED_COLUMN],
        errors="coerce",
    )

    valid_rows = (
        chunk[DATE_COLUMN].notna()
        & (chunk[CANCELLED_COLUMN] == 0)
        & chunk[TARGET_COLUMN].isin([0, 1])
        & chunk[FEATURE_COLUMNS].notna().all(axis=1)
    )

    eligible = chunk.loc[
        valid_rows,
        [DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN],
    ].copy()

    eligible[TARGET_COLUMN] = eligible[TARGET_COLUMN].astype("int8")

    split_masks = _get_split_dates(eligible[DATE_COLUMN])
    counts = {}

    for split_name, mask in split_masks.items():
        split_data = eligible.loc[mask]

        if split_data.empty:
            counts[split_name] = 0
            continue

        split_data.to_csv(
            output_paths[split_name],
            mode="a",
            header=split_name not in written_splits,
            index=False,
        )

        written_splits.add(split_name)
        counts[split_name] = len(split_data)

    return counts


def prepare_splits(
    source_path: str | Path = "data/interim/flight_delays_2024_2025.csv",
    output_dir: str | Path = "data/processed",
    chunksize: int = 200_000,
) -> dict[str, int]:
    """
    Create chronological train, validation, and test datasets.

    Train:      January–December 2024
    Validation: January–March 2025
    Test:       April–June 2025
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)

    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)

    totals = dict.fromkeys(SPLIT_NAMES, 0)
    written_splits: set[str] = set()

    with TemporaryDirectory(dir=output_dir) as temporary_dir:
        temporary_dir = Path(temporary_dir)

        temporary_paths = {name: temporary_dir / f"{name}.csv" for name in SPLIT_NAMES}

        for chunk in pd.read_csv(
            source_path,
            usecols=SOURCE_COLUMNS,
            chunksize=chunksize,
        ):
            counts = _write_chunk(
                chunk,
                temporary_paths,
                written_splits,
            )

            for split_name, count in counts.items():
                totals[split_name] += count

        missing_splits = [name for name, count in totals.items() if count == 0]

        if missing_splits:
            raise ValueError(f"No eligible records found for: {missing_splits}")

        for split_name, temporary_path in temporary_paths.items():
            destination = output_dir / f"{split_name}.csv"
            os.replace(temporary_path, destination)

    return totals
