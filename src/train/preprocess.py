import pickle
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer


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
    bow_path = '../models/mdc1_BoW_Sentiment_Model.pkl'
    pickle.dump(cv, open(bow_path, "wb"))

    return X, y
