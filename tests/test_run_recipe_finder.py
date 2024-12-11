import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.FindingCloseRecipes.run_recipe_finder import reconstruct_pp_recipes, run_recipe_finder


@pytest.fixture
def mock_split_data():
    """
    Fixture pour simuler les données fragmentées utilisées par reconstruct_pp_recipes.
    """
    tags = pd.DataFrame({'id': [1, 2, 3], 'tags': ['tag1', 'tag2', 'tag3']})
    name = pd.DataFrame({'id': [1, 2, 3], 'name': ['name1', 'name2', 'name3']})
    steps = pd.DataFrame({'id': [1, 2, 3], 'steps': ['step1', 'step2', 'step3']})
    ingredients = pd.DataFrame({'id': [1, 2, 3], 'ingredients': ['ing1', 'ing2', 'ing3']})
    numerics = pd.DataFrame({
        'id': [1, 2, 3],
        'calories': [100, 200, 300],
        'log_minutes': [1.0, 2.0, 3.0]
    })

    return {'tags': tags, 'name': name, 'steps': steps, 'ingredients': ingredients, 'numerics': numerics}


@patch("src.FindingCloseRecipes.run_recipe_finder.pd.read_csv")
@patch("src.FindingCloseRecipes.run_recipe_finder.os.path.exists")
def test_reconstruct_pp_recipes(mock_exists, mock_read_csv, mock_split_data):
    """
    Teste la fonction reconstruct_pp_recipes pour s'assurer qu'elle combine correctement les fichiers fragmentés.
    """
    # Simuler l'existence de fichiers CSV et leur contenu
    mock_exists.return_value = True
    mock_read_csv.side_effect = lambda file_path: mock_split_data[file_path.split('_')[-2]]

    # Appeler la fonction
    pp_recipes = reconstruct_pp_recipes()

    # Vérifier que tous les fichiers ont été chargés
    assert not pp_recipes.empty, "Le DataFrame reconstruit ne doit pas être vide."
    assert len(pp_recipes) == 3, "Le DataFrame reconstruit doit contenir 3 lignes."
    assert set(pp_recipes.columns) == {'id', 'tags', 'name', 'steps', 'ingredients', 'calories', 'log_minutes'}, (
        "Le DataFrame reconstruit doit contenir toutes les colonnes fusionnées."
    )

    # Vérifier les appels
    assert mock_read_csv.call_count == 20, "La fonction doit lire 20 fichiers au total (5 colonnes * 4 fichiers)."


@patch("src.FindingCloseRecipes.run_recipe_finder.reconstruct_pp_recipes")
@patch("src.FindingCloseRecipes.run_recipe_finder.RecipeFinder")
def test_run_recipe_finder(mock_recipe_finder, mock_reconstruct, mock_split_data):
    """
    Teste la fonction run_recipe_finder pour s'assurer qu'elle trouve les recettes similaires.
    """
    # Simuler la reconstruction du DataFrame
    mock_reconstruct.return_value = pd.concat(
        [mock_split_data[col] for col in mock_split_data], axis=1
    ).loc[:, ~pd.concat(
        [mock_split_data[col] for col in mock_split_data], axis=1
    ).columns.duplicated()]

    # Simuler le comportement de RecipeFinder
    mock_instance = mock_recipe_finder.return_value
    mock_instance.preprocess.return_value = None
    mock_instance.find_similar_recipes.return_value = pd.DataFrame({
        'id': [2, 3],
        'combined_distance': [0.1, 0.2]
    })

    # Appeler la fonction avec un ID valide
    recipe_id = 1
    result = run_recipe_finder(recipe_id)

    # Vérifier que reconstruct_pp_recipes a été appelé
    mock_reconstruct.assert_called_once()

    # Vérifier que RecipeFinder a été utilisé correctement
    mock_instance.preprocess.assert_called_once()
    mock_instance.find_similar_recipes.assert_called_once_with(recipe_id)

    # Vérifier le résultat
    assert not result.empty, "Le résultat ne doit pas être vide."
    assert list(result['id']) == [2, 3], "Les IDs des recettes similaires ne correspondent pas."
    assert 'combined_distance' in result.columns, "La colonne 'combined_distance' doit être présente."


@patch("src.FindingCloseRecipes.run_recipe_finder.reconstruct_pp_recipes")
def test_run_recipe_finder_invalid_id(mock_reconstruct, mock_split_data):
    """
    Teste la fonction run_recipe_finder avec un ID invalide.
    """
    # Simuler la reconstruction du DataFrame
    mock_reconstruct.return_value = pd.concat(
        [mock_split_data[col] for col in mock_split_data], axis=1
    ).loc[:, ~pd.concat(
        [mock_split_data[col] for col in mock_split_data], axis=1
    ).columns.duplicated()]

    # Appeler la fonction avec un ID invalide
    invalid_id = 999
    with pytest.raises(ValueError, match="Identifiant de recette introuvable."):
        run_recipe_finder(invalid_id)


