import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from src.DataPreprocess.run_preprocessing import *
from src.DataPreprocess.data_preprocessor import DataPreprocessor

@pytest.fixture
def mock_raw_recipes(tmpdir):
    """
    Fixture pour créer un fichier temporaire simulant `Raw_recipes.csv`.
    """
    data = {
        "id": [1, 2, 3],
        "name": ["Recipe A", "Recipe B", "Recipe C"],
        "nutrition": ["[100,10,5,1,2,1,50]", "[200,20,10,2,4,2,100]", "[300,30,15,3,6,3,150]"],
        "ingredients": [["ingr1", "ingr2"], ["ingr3", "ingr4"], ["ingr5", "ingr6"]],
        "tags": [["tag1", "tag2"], ["tag3", "tag4"], ["tag5", "tag6"]],
        "minutes": [30, 60, 90],
    }
    file_path = tmpdir.join("Raw_recipes.csv")
    pd.DataFrame(data).to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def mock_ingr_map(tmpdir):
    """
    Fixture pour créer un fichier temporaire simulant `ingr_map.csv`.
    """
    data = {
        "raw_ingr": ["ingr1", "ingr2", "ingr3"],
        "replaced": ["category1", "category2", "category3"]
    }
    file_path = tmpdir.join("ingr_map.csv")
    pd.DataFrame(data).to_csv(file_path, index=False)
    return str(file_path)

@pytest.fixture
def mock_output_path(tmpdir):
    """
    Fixture pour générer un chemin de fichier de sortie temporaire.
    """
    return str(tmpdir.join("pp_recipes.csv"))

@patch("src.main.DataPreprocessor")
def test_main_success(mock_preprocessor, mock_raw_recipes, mock_ingr_map, mock_output_path):
    """
    Teste le scénario principal où tout se passe bien.
    """
    mock_instance = MagicMock()
    mock_preprocessor.return_value = mock_instance

    # Simuler les appels successifs
    mock_instance.load_data.return_value = None
    mock_instance.preprocess.return_value = pd.DataFrame({"id": [1, 2, 3], "processed": [True, True, True]})
    mock_instance.save_data.return_value = None

    # Chemins simulés
    with patch("src.main.file_path", mock_raw_recipes), \
         patch("src.main.ingredient_map_path", mock_ingr_map), \
         patch("src.main.output_path", mock_output_path):
        main()

    # Vérifier les appels
    mock_preprocessor.assert_called_once_with(mock_raw_recipes, mock_ingr_map)
    mock_instance.load_data.assert_called_once()
    mock_instance.preprocess.assert_called_once()
    mock_instance.save_data.assert_called_once_with(mock_output_path)

def test_main_file_not_found(mock_raw_recipes, mock_ingr_map):
    """
    Teste le comportement lorsque l'un des fichiers requis est manquant.
    """
    invalid_file_path = "invalid/path/to/Raw_recipes.csv"
    with patch("src.main.file_path", invalid_file_path), \
         patch("src.main.ingredient_map_path", mock_ingr_map):
        with pytest.raises(FileNotFoundError, match="Fichier de données introuvable"):
            main()

def test_main_critical_error(mock_raw_recipes, mock_ingr_map):
    """
    Teste le comportement lorsqu'une erreur critique se produit dans le pipeline.
    """
    with patch("src.main.DataPreprocessor.preprocess", side_effect=Exception("Erreur critique")):
        with pytest.raises(Exception, match="Erreur critique"):
            main()
