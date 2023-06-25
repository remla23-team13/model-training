"""Different tests"""
import json

import src.models.train as training
from src.models import evaluation


def test_nondeterminism_robustness() -> None:
    """Train the model with different seeds and check that the accuracy
    is within 5% of the original"""
    model_accuracy = None
    with open("metrics/metrics.json", encoding="utf-8") as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics["accuracy"]
    X, y = training.load_preprocessed_data("data/processed/preprocessed_data.joblib")

    for i in range(20):
        X_train, X_test, y_train, y_test = training.create_split(X, y, random_state=i)
        trained_model = training.train(X_train, y_train)
        metrics = evaluation.test(X_test, y_test, trained_model)
        model_accuracy_ = metrics["accuracy"]
        assert abs(model_accuracy - model_accuracy_) < 0.05
