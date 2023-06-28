"""Test for src/features/preprocess.py"""
import os
from pathlib import Path

import pandas as pd

from src.features.preprocess import preprocess


def test_load_data(data: pd.DataFrame) -> None:
    """Test if data is read as DataFrame"""
    assert isinstance(data, pd.DataFrame)


def test_load_data_columns(data: pd.DataFrame) -> None:
    """Test for expected columns"""
    expected_columns = ["Review", "Liked"]
    assert all(col in data.columns for col in expected_columns)


def test_preprocess_length(
    data: pd.DataFrame, preprocessed_data: list[list[int]]
) -> None:
    """Test data length"""
    assert data.shape[0] == len(preprocessed_data[0])
    assert data.shape[1] == len(preprocessed_data)


def test_write_preprocess_data(data: pd.DataFrame) -> None:
    """Test creation output file"""

    processed_data_path = Path("data/processed/preprocessed_data.joblib")
    if processed_data_path.is_file():
        os.remove(processed_data_path)

    preprocess(data)
    assert processed_data_path.is_file()
