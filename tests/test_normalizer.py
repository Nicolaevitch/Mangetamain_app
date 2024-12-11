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

    with pytest.raises(ValueError, match="Les colonnes spécifiées doivent contenir uniquement des données numériques."):
        normalizer.normalize(mock_data_non_numeric, ["name"])


