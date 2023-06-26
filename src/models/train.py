"""Training phase of the model training pipeline"""
from typing import Any

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

classifier = LinearSVC()


def save_model(model: LinearSVC) -> None:
    """Save the model"""
    print("Saving model..")
    dump(model, "models/model.joblib")


def load_preprocessed_data(preprocessed_data_path: str) -> tuple[list[int], list[int]]:
    """Load the preprocessed data"""
    X, y = load(preprocessed_data_path)
    return X, y


def create_split(X: list[Any], y: list[int], random_state: int = 0) -> Any:
    """Create a train/test split of the data"""
    return train_test_split(X, y, test_size=0.20, random_state=random_state)


def train(
    X_train: list[int], y_train: list[int], model: LinearSVC = classifier
) -> LinearSVC:
    """Train the model"""

    model.fit(X_train, y_train)
    return model


def train_main() -> None:
    """Main function for training"""
    X, y = load_preprocessed_data("data/processed/preprocessed_data.joblib")
    X_train, X_test, y_train, y_test = create_split(X, y)
    model = train(X_train, y_train)
    dump([X_test, y_test], "data/test/test_data.joblib")
    save_model(model)


if __name__ == "__main__":
    train_main()
