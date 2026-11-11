import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

DATA_DIR = Path("data/raw/monthly")
BASE_URL = "https://transtats.bts.gov/PREZIP"

MONTHS = [
    (year, month)
    for year, month_range in (
        (2024, range(1, 13)),
        (2025, range(1, 7)),
    )
    for month in month_range
]

FILE_PREFIX = "On_Time_Reporting_Carrier_On_Time_Performance_1987_present"


def download_data() -> None:
    """Download missing monthly BTS flight datasets."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for year, month in MONTHS:
        filename = f"{year}_{month}.zip"
        destination = DATA_DIR / filename

        if destination.exists():
            try:
                with zipfile.ZipFile(destination) as archive:
                    if archive.testzip() is None:
                        continue
            except zipfile.BadZipFile:
                pass

            destination.unlink()

        url = f"{BASE_URL}/{FILE_PREFIX}_{year}_{month}.zip"
        temporary_file = destination.with_suffix(".tmp")

        print(f"Downloading {filename}...")

        try:
            with urlopen(url, timeout=120) as response:
                temporary_file.write_bytes(response.read())

            with zipfile.ZipFile(temporary_file) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f"Invalid ZIP archive: {filename}")

            temporary_file.replace(destination)

        except (URLError, OSError, zipfile.BadZipFile, ValueError):
            temporary_file.unlink(missing_ok=True)
            raise

    print("All monthly datasets are available.")
