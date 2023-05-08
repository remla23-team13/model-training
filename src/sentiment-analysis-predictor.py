import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import joblib


def get_data():
    return pd.read_csv('a2_RestaurantReviews_FreshDump.tsv',
                       delimiter='\t', quoting=3)


def get_stopwords():
    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    return all_stopwords


def clean_review(review):
    all_stopwords = get_stopwords()
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)
              for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review


def main():
    dataset = get_data()

    corpus = []

    for i in range(0, 100):
        corpus.append(clean_review(dataset['Review'][i]))

    cvFile = 'c1_BoW_Sentiment_Model.pkl'
    # cv = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('./drive/MyDrive/Colab Notebooks/2 Sentiment Analysis (Basic)/3.1 BoW_Sentiment Model.pkl', "rb")))
    cv = pickle.load(open(cvFile, "rb"))

    X_fresh = cv.transform(corpus).toarray()
    X_fresh.shape

    classifier = joblib.load('c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_fresh)

    dataset['predicted_label'] = y_pred.tolist()
    dataset[dataset['predicted_label'] == 1]


if __name__ == "main":
    main()
