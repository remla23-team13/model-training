"""Test for anomalies in feature and label distribution"""

import pandas as pd
import tensorflow_data_validation as tfdv

import src.models.train as training
from src.features.preprocess import load_dataset


def test_feature_distribution() -> None:
    """Test for anomalies in feature distribution"""

    dataset = load_dataset("data/raw/RestaurantReviews.tsv")
    X = dataset.iloc[:, 0].values.tolist()
    y = dataset.iloc[:, 1].values.tolist()
    X_train, X_test, _, _ = training.create_split(X, y)
    training_data = pd.DataFrame(X_train)
    serving_data = pd.DataFrame(X_test)

    training_stats = tfdv.generate_statistics_from_dataframe(training_data)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_data)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats
    )
    assert tfdv.display_anomalies(skew_anomalies) is None


def test_processed_feature_distribution() -> None:
    """Test for anomalies in proprocessed feature distribution"""

    X, y = training.load_preprocessed_data("data/processed/preprocessed_data.joblib")
    X_train, X_test, _, _ = training.create_split(X, y)

    training_data = pd.DataFrame(X_test)
    serving_data = pd.DataFrame(X_train)

    training_stats = tfdv.generate_statistics_from_dataframe(training_data)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_data)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats
    )
    assert tfdv.display_anomalies(skew_anomalies) is None


def test_label_distribution() -> None:
    """Test for anomalies in label feature distribution"""

    dataset = load_dataset("data/raw/RestaurantReviews.tsv")
    X = dataset.iloc[:, 0].values.tolist()
    y = dataset.iloc[:, 1].values.tolist()
    _, _, y_train, y_test = training.create_split(X, y)
    training_labels = pd.DataFrame(y_train)
    serving_labels = pd.DataFrame(y_test)

    training_stats = tfdv.generate_statistics_from_dataframe(training_labels)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_labels)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats
    )
    assert tfdv.display_anomalies(skew_anomalies) is None
