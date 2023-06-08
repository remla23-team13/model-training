import pytest
import json
import train as training

@pytest.fixture
def trained_model():
    from joblib import load
    yield load('models/model.joblib')

def test_nondeterminism_robustness(trained_model):
    model_accuracy = None
    with open('metrics/metrics.json') as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics['accuracy']
    X,y = training.load_preprocessed_data('data/preprocessed_data.joblib')

    for i in range(20):
        X_train, X_test, y_train, y_test = training.create_split(X, y, random_state=i)
        trained_model = training.train(X_train, y_train)
        metrics = training.test(X_test, y_test, trained_model)
        model_accuracy_ = metrics['accuracy']
        assert(abs(model_accuracy - model_accuracy_) < 0.05)


