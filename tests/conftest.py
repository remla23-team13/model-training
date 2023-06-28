"""Config file for pytest where fixtures are defined"""
from typing import Generator

import pandas as pd
import pytest
from joblib import load
from remlalib.preprocess import Preprocess
from sklearn.svm import LinearSVC

from src.features.preprocess import load_dataset
from src.models.train import load_preprocessed_data


@pytest.fixture
def trained_model() -> Generator[LinearSVC, None, None]:
    """Fixture for trained model"""
    yield load("models/model.joblib")


@pytest.fixture(name="data")
def data_() -> Generator[pd.DataFrame, None, None]:
    """Fixture for raw data"""
    yield load_dataset("data/RestaurantReviews.tsv")


@pytest.fixture()
def preprocessed_data() -> Generator[tuple[list[int], list[int]], None, None]:
    """Fixture for processed data"""
    yield load_preprocessed_data("data/processed/preprocessed_data.joblib")


@pytest.fixture()
def preprocessor(data: pd.DataFrame) -> Preprocess:
    """Fixture for processor"""
    preprocess = Preprocess()
    _, _ = preprocess.preprocess_dataset(data)
    yield preprocess
