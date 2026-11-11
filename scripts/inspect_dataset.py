import csv
from collections import Counter

FILE_PATH = "data/raw/extracted/T_ONTIME_REPORTING.csv"


def inspect_dataset():
    missing_counts = Counter()
    cancellation_counts = Counter()
    target_counts = Counter()
    eligible_target_counts = Counter()

    total_rows = 0
    cancelled_rows = 0
    operating_rows = 0
    missing_target_rows = 0

    with open(FILE_PATH, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            for column, value in row.items():
                if value is None or value.strip() == "":
                    missing_counts[column] += 1

            cancelled = row["CANCELLED"].strip()
            target = row["DEP_DEL15"].strip()

            cancellation_counts[cancelled] += 1

            if cancelled == "1.00":
                cancelled_rows += 1
            elif cancelled == "0.00":
                operating_rows += 1

            target_counts[target if target else "<MISSING>"] += 1

            if cancelled == "0.00":
                if target in ("0.00", "1.00"):
                    eligible_target_counts[target] += 1
                else:
                    missing_target_rows += 1

    print("\n========== DATASET SUMMARY ==========")
    print("Total rows:", total_rows)
    print("Cancelled flights:", cancelled_rows)
    print("Operating flights:", operating_rows)

    print("\n========== MISSING VALUES ==========")
    for column, count in sorted(missing_counts.items()):
        if count > 0:
            percentage = count / total_rows * 100
            print(f"{column}: {count} ({percentage:.2f}%)")

    print("\n========== CANCELLATION VALUES ==========")
    for value, count in sorted(cancellation_counts.items()):
        print(f"{value}: {count}")

    print("\n========== RAW TARGET DISTRIBUTION ==========")
    for value, count in sorted(target_counts.items()):
        print(f"{value}: {count}")

    print("\n========== ELIGIBLE TARGET DISTRIBUTION ==========")
    eligible_total = sum(eligible_target_counts.values())

    for value, count in sorted(eligible_target_counts.items()):
        percentage = count / eligible_total * 100 if eligible_total else 0
        print(f"{value}: {count} ({percentage:.2f}%)")

    print("\nMissing/invalid target among operating flights:", missing_target_rows)
    print("Eligible training records:", eligible_total)


if __name__ == "__main__":
    inspect_dataset()
