from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def get_data():
    return pd.read_csv(
        'data/a1_RestaurantReviews_HistoricDump.tsv', delimiter='\t', quoting=3)


def get_stopwords():

    nltk.download('stopwords')

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    return all_stopwords


def get_corpus(dataset):
    all_stopwords = get_stopwords(all_stopwords)
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


def main():
    dataset = get_data()

    corpus = get_corpus(dataset)

    cv = CountVectorizer(max_features=1420)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values

    # Saving BoW dictionary to later use in prediction
    bow_path = 'c1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    accuracy_score(y_test, y_pred)


if __name__ == "main":
    main()
