import pytest
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from src.FindingCloseRecipes.vectorizers import Vectorizer


@pytest.fixture
def sample_text_data():
    """
    Fixture pour simuler une colonne Pandas avec des données textuelles.
    """
    return pd.Series(["This is a test", "Another test case", "More text data"])


@pytest.fixture
def empty_text_data():
    """
    Fixture pour simuler une colonne Pandas vide.
    """
    return pd.Series([])


def test_tfidf_vectorize_valid_data(sample_text_data):
    """
    Teste la méthode tfidf_vectorize avec des données valides.
    """
    # Appeler la méthode
    tfidf_matrix, vectorizer = Vectorizer.tfidf_vectorize(sample_text_data)

    # Vérifier les types de retour
    assert isinstance(tfidf_matrix, type(vectorizer.fit_transform(sample_text_data))), (
        "La matrice TF-IDF retournée doit être une matrice creuse (csr_matrix)."
    )
    assert isinstance(vectorizer, TfidfVectorizer), "Le vectorizer doit être une instance de TfidfVectorizer."

    # Vérifier le contenu de la matrice
    assert tfidf_matrix.shape[0] == len(sample_text_data), (
        "Le nombre de lignes de la matrice TF-IDF doit correspondre au nombre d'entrées dans les données."
    )
    assert tfidf_matrix.shape[1] > 0, "La matrice TF-IDF doit contenir des colonnes."


def test_tfidf_vectorize_empty_data(empty_text_data):
    """
    Teste la méthode tfidf_vectorize avec une colonne vide.
    """
    with pytest.raises(ValueError, match="La colonne fournie est vide."):
        Vectorizer.tfidf_vectorize(empty_text_data)


def test_bow_vectorize_valid_data(sample_text_data):
    """
    Teste la méthode bow_vectorize avec des données valides.
    """
    # Appeler la méthode
    bow_matrix, vectorizer = Vectorizer.bow_vectorize(sample_text_data)

    # Vérifier les types de retour
    assert isinstance(bow_matrix, type(vectorizer.fit_transform(sample_text_data))), (
        "La matrice BoW retournée doit être une matrice creuse (csr_matrix)."
    )
    assert isinstance(vectorizer, CountVectorizer), "Le vectorizer doit être une instance de CountVectorizer."

    # Vérifier le contenu de la matrice
    assert bow_matrix.shape[0] == len(sample_text_data), (
        "Le nombre de lignes de la matrice BoW doit correspondre au nombre d'entrées dans les données."
    )
    assert bow_matrix.shape[1] > 0, "La matrice BoW doit contenir des colonnes."


def test_bow_vectorize_empty_data(empty_text_data):
    """
    Teste la méthode bow_vectorize avec une colonne vide.
    """
    with pytest.raises(ValueError, match="La colonne fournie est vide."):
        Vectorizer.bow_vectorize(empty_text_data)

