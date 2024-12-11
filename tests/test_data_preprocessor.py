import pytest
import pandas as pd
import os
from unittest.mock import patch, MagicMock
from src.DataPreprocess.data_preprocessor import DataPreprocessor


@pytest.fixture
def mock_file_path(tmpdir):
    """
    Fixture pour créer un fichier CSV simulé pour les données brutes.
    """
    file_path = tmpdir.join("raw_data.csv")
    data = {
        "id": [1, 2, 3],
        "minutes": [10, 0, 43201],
        "calories": [500, 10000, 15000],
        "ingredients": [
            "['salt', 'pepper']",
            "['sugar', 'flour']",
            "['butter', 'milk']"
        ],
        "nutrition": [
            "[100, 10, 5, 2, 6, 3, 50]",
            "[200, 20, 10, 4, 12, 6, 100]",
            "[300, 30, 15, 6, 18, 9, 150]"
        ],
        "tags": ["['easy', 'quick']", "['dessert']", "['breakfast']"],
        "steps": ["['step1', 'step2']", "['step3', 'step4']", "['step5', 'step6']"],
        "name": ["Recipe A", "Recipe B", "Recipe C"]
    }
    pd.DataFrame(data).to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def mock_ingredient_map_path(tmpdir):
    """
    Fixture pour créer un fichier CSV simulé pour le mapping des ingrédients.
    """
    file_path = tmpdir.join("ingredient_map.csv")
    mapping_data = {"raw_ingr": ["salt", "sugar", "butter"], "replaced": ["spice", "sweetener", "dairy"]}
    pd.DataFrame(mapping_data).to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def output_path(tmpdir):
    """
    Fixture pour créer un chemin de fichier de sortie simulé.
    """
    return tmpdir.join("output_data.csv")


def test_init_invalid_paths():
    """
    Teste la création de l'instance avec des chemins de fichier invalides.
    """
    with pytest.raises(FileNotFoundError):
        DataPreprocessor("invalid_file.csv", "invalid_map.csv")


def test_load_data(mock_file_path, mock_ingredient_map_path):
    """
    Teste la méthode `load_data`.
    """
    preprocessor = DataPreprocessor(mock_file_path, mock_ingredient_map_path)
    data = preprocessor.load_data()

    # Vérifier que les données sont chargées correctement
    assert isinstance(data, pd.DataFrame), "Les données chargées ne sont pas un DataFrame."
    assert len(data) == 3, "Le nombre de lignes dans les données chargées est incorrect."


def test_save_data(mock_file_path, mock_ingredient_map_path, output_path):
    """
    Teste la méthode `save_data`.
    """
    preprocessor = DataPreprocessor(mock_file_path, mock_ingredient_map_path)
    preprocessor.load_data()
    preprocessor.save_data(output_path)

    # Vérifier que le fichier de sortie est créé
    assert os.path.exists(output_path), "Le fichier de sortie n'a pas été créé."


@patch("src.DataPreprocess.data_preprocessor.DataCleaning")
@patch("src.DataPreprocess.data_preprocessor.FeatEngineering")
@patch("src.DataPreprocess.data_preprocessor.VectorizerPreparator")
@patch("src.DataPreprocess.data_preprocessor.Normalizer")
@patch("src.DataPreprocess.data_preprocessor.DatasetSplitter")
def test_preprocess(
    mock_splitter, mock_normalizer, mock_vectorizer, mock_feat_engineer, mock_cleaner, mock_file_path, mock_ingredient_map_path
):
    """
    Teste la méthode `preprocess` avec des mocks pour les étapes internes.
    """
    mock_cleaner_instance = MagicMock()
    mock_feat_engineer_instance = MagicMock()
    mock_vectorizer_instance = MagicMock()
    mock_normalizer_instance = MagicMock()
    mock_splitter_instance = MagicMock()

    mock_cleaner.return_value = mock_cleaner_instance
    mock_cleaner_instance.get_cleaned_data.return_value = pd.DataFrame({"id": [1, 2], "calories": [500, 800]})

    mock_feat_engineer.return_value = mock_feat_engineer_instance
    mock_feat_engineer_instance.get_preprocessed_data.return_value = pd.DataFrame({"id": [1, 2], "calories": [500, 800]})

    mock_vectorizer.return_value = mock_vectorizer_instance
    mock_vectorizer_instance.get_prepared_data.return_value = pd.DataFrame({"id": [1, 2], "calories": [500, 800]})

    mock_normalizer.return_value = mock_normalizer_instance
    mock_normalizer_instance.normalize.return_value = pd.DataFrame({"id": [1, 2], "calories": [500, 800]})

    mock_splitter.return_value = mock_splitter_instance

    preprocessor = DataPreprocessor(mock_file_path, mock_ingredient_map_path)
    preprocessor.load_data()
    processed_data = preprocessor.preprocess()

    # Vérifier que toutes les étapes ont été appelées
    mock_cleaner.assert_called()
    mock_feat_engineer.assert_called()
    mock_vectorizer.assert_called()
    mock_normalizer.assert_called()
    mock_splitter.assert_called()

    # Vérifier que les données prétraitées sont retournées
    assert isinstance(processed_data, pd.DataFrame), "Les données prétraitées ne sont pas un DataFrame."


def test_preprocess_no_data(mock_ingredient_map_path):
    """
    Teste la méthode `preprocess` sans données chargées.
    """
    preprocessor = DataPreprocessor("dummy_file.csv", mock_ingredient_map_path)
    with pytest.raises(ValueError):
        preprocessor.preprocess()
