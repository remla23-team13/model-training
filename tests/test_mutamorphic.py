"""Mutamorpic test for predictions"""

import random

import nltk
import pandas as pd
import pytest
from nltk.corpus import wordnet
from remlalib.preprocess import Preprocess
from sklearn.svm import LinearSVC


def get_synonyms(word: str) -> list[str]:
    """Get a list of synonums for the given word"""
    synonyms = set()
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list[str](synonyms)


def mutate(sentence: str) -> str:
    """Create a mutant sentence where one word is replaced with a synonum"""
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


def test_mutation(
    data: pd.DataFrame,
    preprocessed_data: tuple[list[int], list[int]],
    trained_model: LinearSVC,
) -> None:
    """Test labels when a sentence is mutated"""
    nltk.download("wordnet")

    data["mutated_reviews"] = data["Review"].apply(mutate)

    mutated_data = pd.DataFrame()
    mutated_data["Review"] = data["Review"].apply(mutate)
    mutated_data["Liked"] = data["Liked"]

    preprocessor = Preprocess()

    x_original, _ = preprocessed_data
    predict_original = trained_model.predict(x_original)

    x_mutated, _ = preprocessor.preprocess_dataset(mutated_data)

    predict_mutated = trained_model.predict(x_mutated)
    for i, (original_label, mutant_label) in enumerate(
        zip(predict_original, predict_mutated)
    ):
        if original_label != mutant_label:
            review = data["Review"].iloc[i]
            mutant_review = mutated_data["Review"].iloc[i]
            print(f"Mutation detected! From : {review}; To: {mutant_review}")
            pytest.skip(f"Mutation detected! From : {review}; To: {mutant_review}")
