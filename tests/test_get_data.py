"""Test the get_data function"""
import os

from src.data.get_data import OUTPUT, get_data


def test_get_data() -> None:
    """Test the get_data function"""
    get_data()
    assert os.path.isfile(OUTPUT)
    os.remove(OUTPUT)
