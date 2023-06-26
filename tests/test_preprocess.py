import pytest
import pandas as pd
from src.preprocess import load_dataset, preprocess
from pathlib import Path
import os

@pytest.fixture()
def data():
    yield load_dataset('data/RestaurantReviews.tsv')

@pytest.fixture()
def preprocessed_data():
    processed_data_path = Path("data/preprocessed_data.joblib")
    if processed_data_path.is_file():
        os.remove(processed_data_path)
    data = load_dataset('data/RestaurantReviews.tsv')
    yield preprocess(data)

def test_load_data(data):
    """Test if data is read as DataFrame"""    
    assert isinstance(data, pd.DataFrame)


def test_load_data_columns(data):
    """Test for expected columns"""
    expected_columns = ["Review", "Liked"]
    assert all(col in data.columns for col in expected_columns)


def test_preprocess_length(data, preprocessed_data):
    """Test data length"""
    assert data.shape[0] == len(preprocessed_data[0])
    assert data.shape[1] == len(preprocessed_data)

def test_write_preprocess_data():
    """Test creation output file"""
    assert Path("data/preprocessed_data.joblib").is_file()

