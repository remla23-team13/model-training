"""Download the dataset from remote location"""
import gdown

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
OUTPUT = "data/RestaurantReviews.tsv"

def get_data():
    """Download data from remote storage"""
    gdown.download(URL, OUTPUT, quiet=False)

if __name__ == "__main__":
    get_data()
