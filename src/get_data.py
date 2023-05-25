import gdown

URL = "https://drive.google.com/uc?id=1VNdztX_xMdfvgAdTdOpVJfHt84YcQxn1&export=download"
output = "data/RestaurantReviews.tsv"

gdown.download(URL, output, quiet=False)