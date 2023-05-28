"""Download the dataset from remote location"""
import gdown

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
OUTPUT = "data/RestaurantReviews.tsv"

gdown.download(URL, OUTPUT, quiet=False)
