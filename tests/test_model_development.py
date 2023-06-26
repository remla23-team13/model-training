
## test model for implicit bias
## model quality on important data slices
## model development
import pytest
import json
from src.train import load_preprocessed_data, create_split, test


@pytest.fixture()
def data():
    yield load_preprocessed_data('data/preprocessed_data.joblib')

@pytest.fixture
def trained_model():
    from joblib import load
    yield load('models/model.joblib')    

def test_model_quality_positive(data, trained_model):
     with open('metrics/metrics.json') as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics['accuracy']
     X,y = data
     _, X_test, _, y_test = create_split(X, y)
     X_postive = [review for (review, label) in zip(X_test, y_test) if label == 1 ]
     y_postive = [1 for label in range(len(X_postive)) ]
     metrics = test(X_postive, y_postive, trained_model)
     metrics = test(X_test, y_test, trained_model)

     model_accuracy_ = metrics['accuracy']

     assert(abs(model_accuracy - model_accuracy_) < 0.1)


def test_model_quality_negative(data, trained_model):
     with open('metrics/metrics.json') as json_file:
        metrics = json.load(json_file)
        model_accuracy = metrics['accuracy']

     X, y = data

     _, X_test, _, y_test = create_split(X, y)
     X_negative = [review for (review, label) in zip(X_test, y_test) if label == 0 ]
     y_negative = [0 for label in range(len(X_negative)) ]
     metrics = test(X_negative, y_negative, trained_model)
     model_accuracy_ = metrics['accuracy']

     assert(abs(model_accuracy - model_accuracy_) < 0.1)
