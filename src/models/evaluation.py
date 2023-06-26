"""Evaluation model metrics"""

import json

from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.svm import LinearSVC


def save_metrics(metrics: dict[str, float]) -> None:
    """Save the metrics"""
    with open("metrics/metrics.json", "w", encoding="utf-8") as outfile:
        json.dump(metrics, outfile)


def test(X: list[int], y: list[int], model: LinearSVC) -> dict[str, float]:
    """Test the model"""

    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall}
    return metrics


if __name__ == "__main__":
    X_test, y_test = load("data/test/test_data.joblib")
    trained_model = load("models/model.joblib")
    metrics_result = test(X_test, y_test, trained_model)
    save_metrics(metrics_result)
