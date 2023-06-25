"""Download the dataset from remote location"""
from urllib.request import urlopen

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
OUTPUT = "data/raw/RestaurantReviews.tsv"


def get_data() -> None:
    """Download the dataset from remote location"""
    with urlopen(URL) as file:
        with open(OUTPUT, "wb") as out:
            out.write(file.read())


if __name__ == "__main__":
    get_data()
