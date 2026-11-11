import csv
import io
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "monthly"
OUTPUT_FILE = PROJECT_ROOT / "data" / "interim" / "flight_delays_2024_2025.csv"

DATE_COLUMN = "FlightDate"
EXPECTED_MONTHS = {
    (year, month)
    for year, months in (
        (2024, range(1, 13)),
        (2025, range(1, 7)),
    )
    for month in months
}


def get_monthly_files() -> list[Path]:
    files = sorted(
        RAW_DATA_DIR.glob("*.zip"),
        key=lambda path: tuple(map(int, path.stem.split("_"))),
    )

    actual_months = {tuple(map(int, path.stem.split("_"))) for path in files}

    if actual_months != EXPECTED_MONTHS:
        missing = sorted(EXPECTED_MONTHS - actual_months)
        unexpected = sorted(actual_months - EXPECTED_MONTHS)
        raise ValueError(
            f"Monthly files do not match the expected period. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    return files


def get_csv_name(archive: zipfile.ZipFile, zip_path: Path) -> str:
    csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]

    if len(csv_files) != 1:
        raise ValueError(f"{zip_path.name} must contain exactly one CSV file.")

    return csv_files[0]


def prepare_dataset() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temp_file = OUTPUT_FILE.with_suffix(".tmp")

    total_rows = 0
    reference_header = None
    date_index = None

    try:
        with temp_file.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)

            for zip_path in get_monthly_files():
                year, month = map(int, zip_path.stem.split("_"))
                month_rows = 0
                first_date = None
                last_date = None

                print(f"Processing {year}-{month:02d}...")

                with zipfile.ZipFile(zip_path) as archive:
                    csv_name = get_csv_name(archive, zip_path)

                    with archive.open(csv_name) as raw_file:
                        text_file = io.TextIOWrapper(
                            raw_file,
                            encoding="utf-8-sig",
                            newline="",
                        )
                        reader = csv.reader(text_file)
                        header = next(reader, None)

                        if not header:
                            raise ValueError(f"Empty CSV header in {zip_path.name}.")

                        while header and not header[-1].strip():
                            header.pop()

                        if reference_header is None:
                            reference_header = header

                            if DATE_COLUMN not in header:
                                raise ValueError(
                                    f"{DATE_COLUMN} is missing from {zip_path.name}."
                                )

                            date_index = header.index(DATE_COLUMN)
                            writer.writerow(header)

                        elif header != reference_header:
                            raise ValueError(f"Column mismatch in {zip_path.name}.")

                        for row_number, row in enumerate(reader, start=2):
                            while (
                                len(row) > len(reference_header) and not row[-1].strip()
                            ):
                                row.pop()

                            if len(row) != len(reference_header):
                                raise ValueError(
                                    f"Unexpected number of columns in "
                                    f"{zip_path.name}, row {row_number}."
                                )

                            flight_date = datetime.strptime(
                                row[date_index].strip(),
                                "%Y-%m-%d",
                            )

                            if (flight_date.year, flight_date.month) != (year, month):
                                raise ValueError(
                                    f"Unexpected date {flight_date.date()} "
                                    f"in {zip_path.name}."
                                )

                            first_date = (
                                flight_date
                                if first_date is None
                                else min(first_date, flight_date)
                            )
                            last_date = (
                                flight_date
                                if last_date is None
                                else max(last_date, flight_date)
                            )

                            writer.writerow(row)
                            month_rows += 1

                if month_rows == 0:
                    raise ValueError(f"No records found in {zip_path.name}.")

                total_rows += month_rows
                print(
                    f"  {month_rows:,} records | "
                    f"{first_date.date()} to {last_date.date()}"
                )

        temp_file.replace(OUTPUT_FILE)

    except Exception:
        temp_file.unlink(missing_ok=True)
        raise

    print("\nDataset preparation completed.")
    print(f"Total records: {total_rows:,}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Size: {OUTPUT_FILE.stat().st_size / (1024**3):.2f} GB")


if __name__ == "__main__":
    prepare_dataset()
