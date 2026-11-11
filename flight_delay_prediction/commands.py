import subprocess
import sys
from pathlib import Path

import pandas as pd
from fire import Fire
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from flight_delay_prediction.data import prepare_splits
from flight_delay_prediction.download import download_data
from flight_delay_prediction.evaluate import evaluate_model
from flight_delay_prediction.predict import predict_flights
from flight_delay_prediction.train import train_model

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config():
    """Load the complete project configuration through Hydra."""
    with initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        return compose(config_name="config")


def pull_dvc(target: str, remote: str) -> bool:
    """Pull an artifact from the configured DVC remote."""
    try:
        subprocess.run(
            ["dvc", "pull", "-r", remote, target],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        print(f"DVC pull failed for {target}.")
        return False


def prepare():
    """Download and prepare the flight datasets."""
    config = load_config()

    download_data()

    script = Path(__file__).resolve().parent.parent / "scripts" / "prepare_dataset.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(Path(config.data.raw_data_dir).resolve()),
            str(Path(config.data.prepared_dataset_path).resolve()),
        ],
        check=True,
    )

    return prepare_splits(
        source_path=config.data.source_path,
        output_dir=config.data.output_dir,
        chunksize=config.data.chunksize,
        train_end=config.data.split_dates.train_end,
        validation_end=config.data.split_dates.validation_end,
        test_end=config.data.split_dates.test_end,
    )


def train():
    """Retrieve data, prepare missing datasets, and train the model."""
    config = load_config()

    pull_dvc("data/processed.dvc", "data_storage")

    required_files = [
        Path(config.training.train_path),
        Path(config.training.validation_path),
    ]

    if not all(path.is_file() for path in required_files):
        prepare()
        config = load_config()

    return train_model(config)


def main():
    """Run the project command-line interface."""
    Fire(
        {
            "prepare": prepare,
            "train": train,
            "evaluate": evaluate,
            "predict": predict,
        }
    )


def evaluate():
    """Evaluate the trained model and compare it with baselines."""
    config = load_config()

    results = evaluate_model(
        test_path=config.training.test_path,
        train_path=config.training.train_path,
        model_path=Path(config.training.model_dir) / "flight_delay_model.json",
        category_path=config.training.category_path,
        output_dir=config.training.plots_dir,
        tracking_uri=config.training.tracking_uri,
        experiment_name=config.training.experiment_name,
        validation_path=config.training.validation_path,
        threshold_selection_metric=config.metrics.threshold_selection_metric,
        config_yaml=OmegaConf.to_yaml(config, resolve=True),
    )

    print(results)


def predict(input_path: str):
    """Generate predictions for flight records."""
    config = load_config()

    model_path = Path(config.training.model_dir) / "flight_delay_model.json"
    category_path = Path(config.training.category_path)

    if not model_path.is_file() or not category_path.is_file():
        pull_dvc("models.dvc", "model_storage")

    if not model_path.is_file() or not category_path.is_file():
        raise FileNotFoundError(
            "Model files are missing. Run 'flight-delay train' first."
        )

    flights = pd.read_csv(input_path)
    results = predict_flights(
        flights,
        model_path=model_path,
        category_path=category_path,
    )

    print(results.head())
    print(f"\nTotal predictions: {len(results)}")


if __name__ == "__main__":
    main()
