import pytest
import pandas as pd
import numpy as np
from src.DataPreprocess.normalizer import Normalizer

@pytest.fixture
def mock_data():
    """
    Fixture pour créer un DataFrame simulé avec des données numériques.
    """
    data = {
        "id": [1, 2, 3],
        "log_minutes": [1.0, 2.0, 3.0],
        "calories": [100, 200, 300],
        "total fat (PDV%)": [10, 20, 30],
        "sugar (PDV%)": [5, 10, 15],
        "sodium (PDV%)": [1, 2, 3],
        "protein (PDV%)": [2, 4, 6],
        "saturated fat (PDV%)": [1, 2, 3],
        "carbohydrates (PDV%)": [50, 100, 150]
    }
    return pd.DataFrame(data)

def test_normalize_valid_columns(mock_data):
    """
    Teste la méthode `normalize` avec des colonnes valides.
    """
    normalizer = Normalizer()
    columns_to_normalize = [
        "log_minutes", "calories", "total fat (PDV%)", "sugar (PDV%)",
        "sodium (PDV%)", "protein (PDV%)", "saturated fat (PDV%)", "carbohydrates (PDV%)"
    ]
    normalized_data = normalizer.normalize(mock_data.copy(), columns_to_normalize)

    # Vérifier que les colonnes sont bien normalisées
    for col in columns_to_normalize:
        assert np.isclose(normalized_data[col].mean(), 0, atol=1e-7), f"La moyenne de la colonne {col} n'est pas 0."
        assert np.isclose(normalized_data[col].std(), 1, atol=1e-7), f"L'écart-type de la colonne {col} n'est pas 1."

def test_normalize_missing_columns(mock_data):
    """
    Teste le comportement de `normalize` lorsque certaines colonnes sont absentes.
    """
    normalizer = Normalizer()
    columns_to_normalize = ["log_minutes", "calories", "missing_column"]

    with pytest.raises(ValueError, match="Colonnes manquantes dans le DataFrame"):
        normalizer.normalize(mock_data.copy(), columns_to_normalize)

def test_normalize_empty_dataframe():
    """
    Teste le comportement de `normalize` avec un DataFrame vide.
    """
    normalizer = Normalizer()
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Colonnes manquantes dans le DataFrame"):
        normalizer.normalize(empty_df, ["log_minutes", "calories"])

def test_normalize_no_numeric_data():
    """
    Teste le comportement de `normalize` avec des colonnes non numériques.
    """
    normalizer = Normalizer()
    mock_data_non_numeric = pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Recipe A", "Recipe B", "Recipe C"]
    })

    with pytest.raises(ValueError, match="Colonnes manquantes dans le DataFrame"):
        normalizer.normalize(mock_data_non_numeric, ["name"])

def test_normalize_partial_columns(mock_data):
    """
    Teste la méthode `normalize` avec un sous-ensemble de colonnes valides.
    """
    normalizer = Normalizer()
    columns_to_normalize = ["log_minutes", "calories"]
    normalized_data = normalizer.normalize(mock_data.copy(), columns_to_normalize)

    # Vérifier que les colonnes spécifiées sont normalisées
    for col in columns_to_normalize:
        assert np.isclose(normalized_data[col].mean(), 0, atol=1e-7), f"La moyenne de la colonne {col} n'est pas 0."
        assert np.isclose(normalized_data[col].std(), 1, atol=1e-7), f"L'écart-type de la colonne {col} n'est pas 1."

    # Vérifier que les autres colonnes ne sont pas modifiées
    for col in mock_data.columns:
        if col not in columns_to_normalize:
            assert normalized_data[col].equals(mock_data[col]), f"La colonne {col} a été modifiée à tort."

