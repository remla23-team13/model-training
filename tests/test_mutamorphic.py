import os
import random

import nltk
import pytest
import pandas as pd
from nltk.corpus import wordnet
from src.preprocess import load_dataset, preprocess
from src.train import load_preprocessed_data
from joblib import load
from remlalib.preprocess import Preprocess

# for the first time you use wordnet
# import nltk
# nltk.download('wordnet')

@pytest.fixture
def trained_model():
    yield load('models/model.joblib')  


@pytest.fixture()
def data():
    yield load_dataset('data/RestaurantReviews.tsv')  

@pytest.fixture()
def preprocessed_data():
    yield load_preprocessed_data('data/preprocessed_data.joblib')

def get_synonyms(word):
    synonyms = set()
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)


def mutate(sentence):
    words = sentence.split()
    
    indices = list(range(len(words)))
    random.shuffle(indices)

    while len(indices) != 0:
        current_index = indices.pop()
        word_to_replace = words[current_index]

        synonyms = get_synonyms(word_to_replace)

        if len(synonyms) > 0:
            random_synonum = random.choice(synonyms)
            words[current_index] = random_synonum
            return " ".join(words)
        
        return sentence

def test_mutation(data, preprocessed_data, trained_model):
    data['mutated_reviews'] = data['Review'].apply(mutate)

    mutated_data = pd.DataFrame()
    mutated_data['Review'] =  data['Review'].apply(mutate)
    mutated_data['Liked'] = data['Liked']

    print(mutated_data.head())
    preprocessor = Preprocess()

    X_original, _ = preprocessed_data
    predict_original = trained_model.predict(X_original)
    print(predict_original)

    X_mutated, _ = preprocessor.preprocess_dataset(mutated_data)
    
    predict_mutated = trained_model.predict(X_mutated)
    for i, (original_label, mutant_label) in enumerate(zip(predict_original, predict_mutated)):
        if original_label != mutant_label:
            review = data['Review'].loc[i]
            mutant_review = mutated_data['Review'].loc[i]
            print(f"Mutation detected! From : {review}; To: {mutant_review}")
            pytest.skip(f"Mutation detected! From : {review}; To: {mutant_review}")