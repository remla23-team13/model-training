import pandas as pd


def load_data(data_path):
    print("Loading data...")
    return pd.read_csv(
        data_path, delimiter='\t', quoting=3)
