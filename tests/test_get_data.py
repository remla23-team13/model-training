"""Test for src/data/get_data.py"""
from pathlib import Path

import psutil
from pytest import FixtureRequest

from src.data.get_data import OUTPUT, get_data


def test_get_data() -> None:
    """Test that output file exists"""
    get_data()
    assert Path(OUTPUT).is_file()


def test_memory_usage(request: FixtureRequest) -> None:
    """Test memory usage when getting data"""

    process = psutil.Process()
    memory_info_start = process.memory_info()  # Initial memory usage
    memory_usage_start = memory_info_start.rss / (1024 * 1024)  # Convert to megabytes

    get_data()

    def finalizer() -> None:
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024)  # Convert to megabytes

        assert memory_usage_start - memory_usage < 50

    request.addfinalizer(finalizer)
