# Flight Delay Prediction

## Project Overview

This project predicts whether a domestic US flight departs at least 15 minutes late, using data available before the scheduled departure.

The project uses the BTS On-Time Performance dataset, and binary classification is done using XGBoost.

## Dataset Description

### Data Source

The project utilizes the dataset, ‘‘Reporting Carrier On-Time Performance’’ published by the US Bureau of Transportation Statistics (BTS).

The dataset comprises records of domestic flights operated by the US air carriers. Each record represents a single flight, and details such as the airline, the airports of origin and destination, scheduled flight, actual flight, and delays at departure and arrival are included.

For this project, 18 monthly datasets, January 2024 through June 2025, were collected.

The monthly datasets are downloaded from the BTS TranStats website as ZIP archives. Each ZIP archive contains a single CSV file with flight-level records and 109 columns.

### Dataset Overview

| Property                             | Description                               |
| ------------------------------------ | ----------------------------------------- |
| Dataset                              | BTS Reporting Carrier On-Time Performance |
| Source                               | US Bureau of Transportation Statistics    |
| Coverage                             | January 2024 - June 2025                  |
| Number of monthly files              | 18                                        |
| Original records                     | 10,525,737                                |
| Original columns                     | 109                                       |
| Eligible records after preprocessing | 10,377,788                                |
| Selected model features              | 8                                         |
| Prediction target                    | `DepDel15`                                |

### Target Variable

The target variable `DepDel15` indicates whether a flight departed 15 minutes or more late.

`DepDel15` =

- 0 if flight departed less than 15 minutes late
- 1 if flight departed 15 minutes or more late.

Therefore this is a binary classification problem.

### Selected Features

Eights features have been selected from the original dataset:

| Feature           | Description                                     |
| ----------------- | ----------------------------------------------- |
| DayOfWeek         | Day of week of flight schedule                  |
| Reporting_Airline | Code of reporting airline                       |
| OriginAirportID   | Unique ID of origin airport                     |
| DestAirportID     | Unique ID of destination airport                |
| CRSDepTime        | Scheduled departure time                        |
| CRSArrTime        | Scheduled arrival time                          |
| CRSElapsedTime    | Scheduled flight time                           |
| Distance          | Distance between origin and destination airport |

These features are selected as they represent data prior to scheduled flight departure.

### Data Preprocessing

18 monthly csv files are combined in a streaming process to form a single large dataset, as loading all the files simultaneously in memory was not feasible.

The pre-processing steps included:

1. Removing flights that were cancelled (Cancelled = Yes)
2. Removing records with invalid or missing values for `DepDel15`
3. Removing records with missing values for selected features of the model
4. Removing records with invalid binary values for target
5. Arranging records as per FlightDate in ascending order
6. Retaining selected features of the model and the target for further analysis

The dataset is now ready for modeling.

After cleaning the data, the total number of flight records was 10,377,788.

The processed dataframe has the following distribution for the target variable:

| Departure status        |    Records | Percentage |
| ----------------------- | ---------: | ---------: |
| On time (0)             |  8,221,548 |    79.22 % |
| 15+ minutes delayed (1) |  2,156,240 |    20.78 % |
| Total                   | 10,377,788 |      100 % |

### Dataset Splitting

Chronological instead of random, was used for splitting the data to assess the model's future ability to classify flight records.

| Dataset    | Time Period       |    Records |
| ---------- | ----------------- | ---------: |
| Training   | Jan-Dec 2024      |  6,982,746 |
| Validation | Jan-Mar 2025      |  1,614,863 |
| Testing    | Apr-Jun 2025      |  1,780,179 |
| Total      | Jan 2024-Jun 2025 | 10,377,788 |

The training set was used to fit the model, the validation set was used for model selection and classification threshold, and the test set was used to evaluate the final model.

### Prevention of Data Leakage

Only data available prior to departure is used in prediction.

Actual departure times, actual delay times, taxi-out times, wheels-off times, and all other data generated during or after flight operations are not used.

This is to ensure that the model is not learning to use data that is unavailable at the time the prediction is intended to be made.

## Data Splitting

The dataset is split in a chronological fashion:

| Dataset    | Period                |
| ---------- | --------------------- |
| Training   | January-December 2024 |
| Validation | January-March 2025    |
| Testing    | April-June 2025       |

## Model and Evaluation

XGBoost's training API with support for categorical features is used in the project.

The validation dataset is used for model selection and F1-based threshold selection. The final model is evaluated against the untouched test dataset.

The evaluation metrics are:

- PR-AUC
- ROC-AUC
- Precision
- Recall
- F1-Score
- Confusion matrix

Baselines using the majority class and historical airline-level delay rates are also evaluated.

## Project Structure

```text
configs/
    Hydra configuration files

data/
    Raw and processed datasets

flight_delay_prediction/
    Python package

models/
    Trained models

plots/
    Evaluation plots

scripts/
    Datasets preparation and inspection scripts

tests/
    Unit tests
```

## Setup

Requirements: Python 3.12 and UV.

Clone the repository and install dependencies:

```bash
uv sync --all-groups
```

Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Train

Training the model:

```bash
uv run flight-delay train
```

Model evaluation:

```bash
uv run flight-delay evaluate
```

Generating predictions:

```bash
uv run flight-delay predict --input_path data/processed/test.csv
```

Testing the package:

```bash
uv run pytest
```

Code quality check:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pre-commit run --all-files
```

## MLOps tools

- UV - dependency and environment management
- Hydra - configuration management
- DVC - dataset/model artifacts tracking
- MLflow - experiment tracking and logging evaluation
- XGBoost - model training
- Pytest and Ruff - tests and code quality checks

## Reproducibility

The repository keeps DVC metadata instead of storing big datasets and models in Git directly. The project provides monthly BTS data downloader and dataset preparing scripts.

Remote DVC storage should be set up for the environment that runs the project. Local filesystem remotes which were used during development are machine-specific.
