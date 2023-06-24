"""Preprocessing phase of the model training pipeline"""
from typing import Any

import pandas as pd
from joblib import dump
from remlalib.preprocess import Preprocess

preprocessor = Preprocess()


def load_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from data_path"""
    return pd.read_csv(data_path, delimiter="\t", quoting=3)


def preprocess(dataset: pd.DataFrame) -> tuple[Any, Any]:
    """Preprocess the dataset and save it"""

    X, y = preprocessor.preprocess_dataset(dataset)

    preprocessed_data_path = "data/preprocessed_data.joblib"
    dump([X, y], preprocessed_data_path)
    return X, y


def main() -> None:
    """Load the dataset and preprocess it"""
    dataset = load_dataset("data/RestaurantReviews.tsv")
    preprocess(dataset)


if __name__ == "__main__":
    main()
