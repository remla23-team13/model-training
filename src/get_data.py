"""Download the dataset from remote location"""
import gdown

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
OUTPUT = "data/RestaurantReviews.tsv"


if __name__ == "__main__":
    gdown.download(URL, OUTPUT, quiet=False)
