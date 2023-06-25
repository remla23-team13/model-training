"""Download the dataset from remote location"""
from urllib.request import urlopen

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
OUTPUT = "data/RestaurantReviews.tsv"

if __name__ == "__main__":
    with urlopen(URL) as file:
        with open(OUTPUT, "wb") as out:
            out.write(file.read())
