"""Test model quality on data slices"""

## test model for implicit bias
## model quality on important data slices
## model development
import json

from sklearn.svm import LinearSVC

from src.models import evaluation, train


def test_model_quality_positive(
    preprocessed_data: tuple[list[int], list[int]], trained_model: LinearSVC
) -> None:
    """Test model quality on positive samples"""

    with open("metrics/metrics.json", encoding="utf-8") as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics["accuracy"]

    print(model_accuracy)

    X, y = preprocessed_data

    _, X_test, _, y_test = train.create_split(X, y)
    x_postive = [review for (review, label) in zip(X_test, y_test) if label == 1]
    y_postive = [1 for _ in range(len(x_postive))]
    metrics = evaluation.test(x_postive, y_postive, trained_model)

    model_accuracy_ = metrics["accuracy"]

    assert abs(model_accuracy - model_accuracy_) < 0.1


def test_model_quality_negative(
    preprocessed_data: tuple[list[int], list[int]], trained_model: LinearSVC
) -> None:
    """Test model quality on negative samples"""

    with open("metrics/metrics.json", encoding="utf-8") as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics["accuracy"]

    X, y = preprocessed_data

    _, X_test, _, y_test = train.create_split(X, y)
    x_negative = [review for (review, label) in zip(X_test, y_test) if label == 0]
    y_negative = [0 for _ in range(len(x_negative))]
    metrics = evaluation.test(x_negative, y_negative, trained_model)

    model_accuracy_ = metrics["accuracy"]

    assert abs(model_accuracy - model_accuracy_) < 0.1
