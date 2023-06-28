"""Test for the whole pipeline from data gathering to prediction"""
from pathlib import Path

from joblib import load

from src.data import get_data
from src.features import preprocess
from src.models import train


def test_integration() -> None:
    """Integration test"""
    get_data.get_data()
    preprocess.main()
    train.train_main()

    data_path = Path("data/RestaurantReviews.tsv")
    processed_data_path = Path("data/processed/preprocessed_data.joblib")
    model_path = Path("models/model.joblib")
    metric_path = Path("metrics/metrics.json")

    assert data_path.is_file()
    assert processed_data_path.is_file()
    assert model_path.is_file()
    assert metric_path.is_file()

    model = load("models/model.joblib")

    postive_sample = preprocess.preprocessor.preprocess_sample("That was great")
    negative_sample = preprocess.preprocessor.preprocess_sample("That was bad")
    y_pred = model.predict([postive_sample, negative_sample])
    assert y_pred[0] == 1
    assert y_pred[1] == 0
