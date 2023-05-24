from train.load_data import load_data
from train.preprocess import preprocess
from train.save_model import save_model

if __name__ == "__main__":

    dataset = load_data('../data/a1_RestaurantReviews_HistoricDump.tsv')
    X, y = preprocess(dataset)
    save_model(X, y)
    print("Finised!")
