import pickle
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from joblib import dump

def load_dataset(data_path):
    return pd.read_csv(data_path, delimiter='\t', quoting=3)

def get_stopwords():
    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    return all_stopwords


def get_corpus(dataset):
    all_stopwords = get_stopwords()
    corpus = []

    ps = PorterStemmer()

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word)
                  for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def preprocess(dataset):
    print("Preprocessing data...")
    corpus = get_corpus(dataset)

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Saving BoW dictionary to later use in prediction
    bow_path = 'preprocessor/preprocessor.joblib'
    dump(cv, bow_path)
    preprocessed_data_path = 'data/preprocessed_data.joblib'
    dump(X, preprocessed_data_path)
    return X, y


def main():
    dataset = load_dataset('data/RestaurantReviews.tsv')
    preprocess(dataset)

if __name__ == "__main__":
    main()