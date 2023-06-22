"""Preprocessing phase of the model training pipeline"""
import re

import nltk
import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


def load_dataset(data_path):
    """Load dataset from data_path"""
    return pd.read_csv(data_path, delimiter="\t", quoting=3)


def get_stopwords():
    """Obtain the list of stopwords"""
    nltk.download("stopwords")

    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")

    return all_stopwords


def get_corpus(dataset):
    """produce the corpus from the dataset by applying preprocessing steps"""
    all_stopwords = get_stopwords()
    corpus = []

    porter_stemmer = PorterStemmer()

    for i in range(0, 900):
        review = re.sub("[^a-zA-Z]", " ", dataset["Review"][i])
        review = review.lower()
        review = review.split()
        review = [
            porter_stemmer.stem(word)
            for word in review
            if not word in set(all_stopwords)
        ]
        review = " ".join(review)
        corpus.append(review)
    return corpus


def preprocess(dataset):
    """Preprocess the dataset and save it"""

    print("Preprocessing data...")
    corpus = get_corpus(dataset)

    count_vectorizer = CountVectorizer(max_features=1420)
    X = count_vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Saving BoW dictionary to later use in prediction
    bow_path = "preprocessor/preprocessor.joblib"
    dump(count_vectorizer, bow_path)
    preprocessed_data_path = "data/preprocessed_data.joblib"
    dump(X, preprocessed_data_path)
    return X, y


def main():
    """Load the dataset and preprocess it"""
    dataset = load_dataset("data/RestaurantReviews.tsv")
    preprocess(dataset)


if __name__ == "__main__":
    main()
