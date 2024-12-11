import pytest
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.FindingCloseRecipes.distances import DistanceCalculator


@pytest.fixture
def mock_numeric_data():
    """
    Fixture pour simuler un DataFrame de caractéristiques numériques.
    """
    data = {
        "feature1": [1.0, 2.0, 3.0],
        "feature2": [4.0, 5.0, 6.0],
        "feature3": [7.0, 8.0, 9.0],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_sparse_data():
    """
    Fixture pour simuler une matrice TF-IDF creuse et les mappings ID-index.
    """
    tfidf_matrix = csr_matrix([
        [0.1, 0.2, 0.3],
        [0.0, 0.5, 0.6],
        [0.7, 0.8, 0.0]
    ])
    id_to_index = pd.Series({101: 0, 102: 1, 103: 2})
    index_to_id = pd.Series({0: 101, 1: 102, 2: 103})
    return tfidf_matrix, id_to_index, index_to_id


def test_euclidean_distance(mock_numeric_data):
    """
    Teste la méthode euclidean_distance avec des données numériques simulées.
    """
    weights = np.array([0.5, 0.3, 0.2])  # Poids pour chaque colonne
    recipe_index = 0

    distances = DistanceCalculator.euclidean_distance(
        numeric_df=mock_numeric_data,
        recipe_index=recipe_index,
        weights_array=weights
    )

    expected_distances = np.array([0.0, 1.44913767, 2.89827535])  # Distances attendues
    assert np.allclose(distances, expected_distances, atol=1e-6), "Les distances euclidiennes sont incorrectes."


def test_euclidean_distance_invalid_index(mock_numeric_data):
    """
    Teste la méthode euclidean_distance avec un index invalide.
    """
    weights = np.array([0.5, 0.3, 0.2])

    with pytest.raises(ValueError, match="Index de recette invalide."):
        DistanceCalculator.euclidean_distance(
            numeric_df=mock_numeric_data,
            recipe_index=5,  # Index invalide
            weights_array=weights
        )


def test_cosine_distance_sparse(mock_sparse_data):
    """
    Teste la méthode cosine_distance_sparse avec des données simulées.
    """
    tfidf_matrix, id_to_index, index_to_id = mock_sparse_data
    recipe_id = 101

    distances = DistanceCalculator.cosine_distance_sparse(
        recipe_id=recipe_id,
        tfidf_matrix=tfidf_matrix,
        id_to_index=id_to_index,
        index_to_id=index_to_id
    )

    expected_distances = np.array([0.0, 0.03453002, 0.22222222])  # Distances attendues
    assert np.allclose(distances, expected_distances, atol=1e-6), "Les distances cosinus sont incorrectes."


def test_cosine_distance_sparse_invalid_id(mock_sparse_data):
    """
    Teste la méthode cosine_distance_sparse avec un ID de recette invalide.
    """
    tfidf_matrix, id_to_index, index_to_id = mock_sparse_data
    recipe_id = 999  # ID invalide

    with pytest.raises(ValueError, match="Identifiant de recette introuvable."):
        DistanceCalculator.cosine_distance_sparse(
            recipe_id=recipe_id,
            tfidf_matrix=tfidf_matrix,
            id_to_index=id_to_index,
            index_to_id=index_to_id
        )

