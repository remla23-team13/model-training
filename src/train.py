"""Training phase of the model training pipeline"""
import json

from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from preprocess import load_dataset


classifier = LinearSVC()

def save_metrics(metrics):
    """Save the metrics"""
    with open('metrics/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)

def save_model(model):
    """Save the model"""
    print("Saving model..")
    dump(model, 'models/model.joblib')

def load_preprocessed_data(preprocessed_data_path):
    dataset = load_dataset('data/RestaurantReviews.tsv')
    X = load(preprocessed_data_path)
    y = dataset.iloc[:, -1].values
    return X, y

def create_split(X, y, random_state=0):
    """Create a train/test split of the data"""
    return train_test_split(
        X, y, test_size=0.20, random_state=random_state)

def train(X_train, y_train, classifier=classifier):
    """Train the model"""

    classifier.fit(X_train, y_train)
    return classifier

def test(X_test, y_test, model):
    """Test the model"""

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }
    return metrics

if __name__ == "__main__":
    X, y = load_preprocessed_data('data/preprocessed_data.joblib')
    X_train, X_test, y_train, y_test = create_split(X, y)
    model = train(X_train, y_train)
    metrics = test(X_test, y_test, model)
    save_metrics(metrics)
    save_model(model)
