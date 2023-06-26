from src.get_data import get_data, OUTPUT
from pathlib import Path
import psutil


def test_get_data():
    get_data()
    assert Path(OUTPUT).is_file()


def test_memory_usage(request):
    """Test memory usage when getting data """
    process = psutil.Process()
    memory_info_start = process.memory_info()  # Initial memory usage
    memory_usage_start = memory_info_start.rss / (1024 * 1024)  # Convert to megabytes

    get_data()

    def finalizer():
        memory_info = process.memory_info()
        memory_usage = memory_info.rss / (1024 * 1024)  # Convert to megabytes

        assert memory_usage_start - memory_usage < 50 

    request.addfinalizer(finalizer)
