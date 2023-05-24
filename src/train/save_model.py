from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def save_model(X, y):
    print("Training model..")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:", cm)

    accuracy = accuracy_score(y_test, y_pred)

    print("Accuracy: ", accuracy)

    print("Saving model..")
    # Exporting NB Classifier to later use in prediction
    joblib.dump(classifier, '../models/c2_Classifier_Sentiment_Model')
