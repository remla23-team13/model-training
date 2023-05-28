"""Training phase of the model training pipeline"""
from joblib import dump, load
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from preprocess import load_dataset
import json

def train():
    """Train the model and save it"""
    print("Training model..")

    dataset = load_dataset('data/RestaurantReviews.tsv')
    X = load('data/preprocessed_data.joblib')
    y = dataset.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

    with open('metrics/metrics.json', 'w') as outfile:
        json.dump(metrics, outfile)

    print("Saving model..")

    dump(classifier, 'models/model.joblib')


if __name__ == "__main__":
    train()
