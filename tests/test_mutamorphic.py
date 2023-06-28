"""Mutamorpic test for predictions"""

import random

import nltk
import pandas as pd
from nltk.corpus import wordnet
from remlalib.preprocess import Preprocess
from sklearn.svm import LinearSVC


def test_mutation(
    data: pd.DataFrame,
    preprocessor: Preprocess,
    trained_model: LinearSVC,
) -> None:
    """Test labels when a sentence is mutated"""
    nltk.download("wordnet")

    x_original = [preprocessor.preprocess_sample(row[0]) for row in data.values]
    predict_original = trained_model.predict(x_original)

    mutated_reviews = [
        mutate_consistently(row[0], trained_model, preprocessor) for row in data.values
    ]
    x_mutated = [preprocessor.preprocess_sample(review) for review in mutated_reviews]
    predict_mutated = trained_model.predict(x_mutated)

    # make sure mutations have actually been applied
    assert not (data["Review"].values == mutated_reviews).all()

    # make sure all mutations are consistent
    assert (predict_mutated == predict_original).all()


def get_synonyms(word: str) -> list[str]:
    """Get a list of synonums for the given word"""
    synonyms = set()
    for synonym in wordnet.synsets(word):
        for lemma in synonym.lemmas():
            synonyms.add(lemma.name())
    if word in synonyms:
        synonyms.remove(word)
    return list[str](synonyms)


def mutate(words: list[str], already_tried: list[str]) -> tuple[str, list[str]]:
    """Create a mutant sentence where one word is replaced with a synonum"""

    indices = [i for i, word in enumerate(words) if word not in already_tried]
    random.shuffle(indices)

    attempted_words = []

    while len(indices) != 0:
        current_index = indices.pop()
        word_to_replace = words[current_index]

        synonyms = get_synonyms(word_to_replace)
        attempted_words.append(word_to_replace)

        if len(synonyms) > 0:
            random_synonum = random.choice(synonyms)
            words[current_index] = random_synonum
            break

    return " ".join(words), attempted_words


def mutate_consistently(
    original_sentence: str, trained_model: LinearSVC, preprocessor: Preprocess
) -> str:
    """Try to find mutant that results in same label, if not found return original sentence"""

    attempted_words: list[str] = []
    attempt_count = 0
    words = original_sentence.split()

    processed_original = preprocessor.preprocess_sample(original_sentence)
    original_label = trained_model.predict([processed_original])

    while len(attempted_words) <= len(words) and attempt_count < len(words):
        words = original_sentence.split()
        mutated_sentence, attempts = mutate(words, attempted_words)
        attempted_words += attempts
        sample = preprocessor.preprocess_sample(mutated_sentence)
        y_pred = trained_model.predict([sample])

        # check if consistent
        if y_pred[0] == original_label:
            return mutated_sentence

        attempt_count += 1
    print(f"No mutant that preserves the label can be found for {original_sentence}")
    return original_sentence
