import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.FindingCloseRecipes.recipe_finder import RecipeFinder


@pytest.fixture
def mock_recipes_df():
    """
    Fixture pour simuler un DataFrame contenant les recettes.
    """
    data = {
        'id': [101, 102, 103],
        'name': ['Recipe A', 'Recipe B', 'Recipe C'],
        'tags': ['tag1 tag2', 'tag2 tag3', 'tag3 tag4'],
        'steps': ['step1 step2', 'step2 step3', 'step3 step4'],
        'ingredients': ['ing1 ing2', 'ing2 ing3', 'ing3 ing4'],
        'log_minutes': [1.0, 2.0, 3.0],
        'calories': [100, 200, 300],
        'total fat (PDV%)': [10, 20, 30],
        'sugar (PDV%)': [5, 10, 15],
        'sodium (PDV%)': [1, 2, 3],
        'protein (PDV%)': [2, 4, 6],
        'saturated fat (PDV%)': [1, 2, 3],
        'carbohydrates (PDV%)': [50, 100, 150],
    }
    return pd.DataFrame(data)


@patch("src.FindingCloseRecipes.vectorizers.Vectorizer.tfidf_vectorize")
@patch("src.FindingCloseRecipes.vectorizers.Vectorizer.bow_vectorize")
def test_preprocess(mock_bow_vectorize, mock_tfidf_vectorize, mock_recipes_df):
    """
    Teste la méthode preprocess pour vérifier que les matrices TF-IDF et BoW sont générées correctement.
    """
    # Mock des sorties des vectorizers
    mock_tfidf_vectorize.return_value = (np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), None)
    mock_bow_vectorize.return_value = (np.array([[1, 0], [0, 1], [1, 1]]), None)

    # Initialiser RecipeFinder et appeler preprocess
    recipe_finder = RecipeFinder(mock_recipes_df)
    recipe_finder.preprocess()

    # Vérifier les appels des méthodes vectorizer
    assert mock_tfidf_vectorize.call_count == 3, "Les colonnes 'name', 'tags' et 'steps' doivent être vectorisées."
    assert mock_bow_vectorize.call_count == 1, "La colonne 'ingredients' doit être vectorisée."

    # Vérifier que les matrices ont été assignées
    assert recipe_finder.tfidf_name is not None, "La matrice TF-IDF pour 'name' doit être créée."
    assert recipe_finder.tfidf_tags is not None, "La matrice TF-IDF pour 'tags' doit être créée."
    assert recipe_finder.tfidf_steps is not None, "La matrice TF-IDF pour 'steps' doit être créée."
    assert recipe_finder.bow_ingredients is not None, "La matrice BoW pour 'ingredients' doit être créée."


@patch("src.FindingCloseRecipes.distances.DistanceCalculator.cosine_distance_sparse")
@patch("src.FindingCloseRecipes.distances.DistanceCalculator.euclidean_distance")
def test_find_similar_recipes(mock_euclidean_distance, mock_cosine_distance_sparse, mock_recipes_df):
    """
    Teste la méthode find_similar_recipes pour vérifier la combinaison des distances et la récupération des recettes similaires.
    """
    # Mock des distances
    mock_cosine_distance_sparse.return_value = np.array([0.0, 0.2, 0.4])
    mock_euclidean_distance.return_value = np.array([0.0, 0.3, 0.5])

    # Initialiser RecipeFinder et appeler preprocess
    recipe_finder = RecipeFinder(mock_recipes_df)
    recipe_finder.preprocess()

    # Appeler find_similar_recipes
    similar_recipes = recipe_finder.find_similar_recipes(101)

    # Vérifier que les fonctions de distance sont appelées
    assert mock_cosine_distance_sparse.call_count == 4, "Les distances cosinus doivent être calculées pour 4 colonnes."
    assert mock_euclidean_distance.call_count == 1, "La distance euclidienne doit être calculée pour les colonnes numériques."

    # Vérifier le résultat
    assert not similar_recipes.empty, "Le résultat ne doit pas être vide."
    assert list(similar_recipes['id']) == [102, 103], "Les IDs des recettes similaires ne correspondent pas."
    assert 'combined_distance' in similar_recipes.columns, "La colonne 'combined_distance' doit être présente."


def test_find_similar_recipes_invalid_id(mock_recipes_df):
    """
    Teste la méthode find_similar_recipes avec un ID de recette invalide.
    """
    recipe_finder = RecipeFinder(mock_recipes_df)
    recipe_finder.preprocess()

    # Vérifier qu'une exception est levée pour un ID invalide
    with pytest.raises(ValueError, match="Identifiant de recette introuvable."):
        recipe_finder.find_similar_recipes(999)
