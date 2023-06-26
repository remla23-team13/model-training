
## test shape of distribution of features, should be the same in training and serving data
import pandas as pd
import tensorflow_data_validation as tfdv
from src.preprocess import load_dataset
from src.train import load_preprocessed_data, create_split

def test_feature_distribution():
    dataset = load_dataset('data/RestaurantReviews.tsv')
    X = dataset.iloc[:, 0].values 
    y = dataset.iloc[:, 1].values
    X_train, X_test, _, _ = create_split(X, y)
    training_data = pd.DataFrame(X_train)
    serving_data = pd.DataFrame(X_test)

    training_stats = tfdv.generate_statistics_from_dataframe(training_data)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_data)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats)
    assert tfdv.display_anomalies(skew_anomalies) == None

def test_processed_feature_distribution():
    X, y = load_preprocessed_data('data/preprocessed_data.joblib')
    X_train, X_test, _, _ = create_split(X, y)

    training_data = pd.DataFrame(X_test)
    serving_data = pd.DataFrame(X_train)

    training_stats = tfdv.generate_statistics_from_dataframe(training_data)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_data)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats)
    assert tfdv.display_anomalies(skew_anomalies) == None

def test_label_distribution():
    dataset = load_dataset('data/RestaurantReviews.tsv')
    X = dataset.iloc[:, 0].values 
    y = dataset.iloc[:, 1].values
    _, _, y_train, y_test = create_split(X, y)
    training_labels = pd.DataFrame(y_train)
    serving_labels = pd.DataFrame(y_test)

    training_stats = tfdv.generate_statistics_from_dataframe(training_labels)
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_labels)

    schema = tfdv.infer_schema(statistics=training_stats)

    skew_anomalies = tfdv.validate_statistics(
        statistics=training_stats, schema=schema, serving_statistics=serving_stats)
    assert tfdv.display_anomalies(skew_anomalies) == None



    



