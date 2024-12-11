import pytest
import pandas as pd
import numpy as np
from src.DataPreprocess.feat_engineering import FeatEngineering

@pytest.fixture
def mock_data():
    """
    Fixture pour créer un DataFrame simulé avec des données brutes.
    """
    data = {
        "id": [1, 2, 3],
        "nutrition": [
            "[100, 10, 5, 2, 6, 3, 50]",
            "[200, 20, 10, 4, 12, 6, 100]",
            "[300, 30, 15, 6, 18, 9, 150]"
        ],
        "minutes": [10, 30, 0],
        "submitted": ["2020-01-01", "2020-01-02", "2020-01-03"],
        "description": ["desc1", "desc2", "desc3"],
        "n_steps": [5, 10, 15],
        "n_ingredients": [3, 5, 7]
    }
    return pd.DataFrame(data)


def test_init_invalid_data():
    """
    Teste si une exception est levée lorsque le paramètre `data` n'est pas un DataFrame.
    """
    with pytest.raises(TypeError):
        FeatEngineering(data="not_a_dataframe")


def test_extract_nutrition_features(mock_data):
    """
    Teste la méthode `extract_nutrition_features`.
    """
    fe = FeatEngineering(mock_data)
    fe.extract_nutrition_features()
    
    # Vérifier que les colonnes nutritionnelles sont créées
    expected_columns = [
        "calories", "total fat (PDV%)", "sugar (PDV%)", "sodium (PDV%)",
        "protein (PDV%)", "saturated fat (PDV%)", "carbohydrates (PDV%)"
    ]
    assert all(col in fe.data.columns for col in expected_columns), "Les colonnes nutritionnelles ne sont pas toutes présentes."

    # Vérifier les valeurs des colonnes créées
    assert fe.data["calories"].iloc[0] == 100.0, "La valeur de 'calories' est incorrecte."
    assert fe.data["carbohydrates (PDV%)"].iloc[2] == 150.0, "La valeur de 'carbohydrates (PDV%)' est incorrecte."


def test_extract_nutrition_features_missing_column(mock_data):
    """
    Teste le comportement de `extract_nutrition_features` lorsque la colonne 'nutrition' est absente.
    """
    mock_data.drop(columns=["nutrition"], inplace=True)
    fe = FeatEngineering(mock_data)

    with pytest.raises(KeyError):
        fe.extract_nutrition_features()


def test_drop_useless_features(mock_data):
    """
    Teste la méthode `drop_useless_features`.
    """
    fe = FeatEngineering(mock_data)
    fe.drop_useless_features()

    # Vérifier que les colonnes inutiles sont supprimées
    dropped_columns = ["submitted", "nutrition", "description", "n_steps", "n_ingredients"]
    assert all(col not in fe.data.columns for col in dropped_columns), "Certaines colonnes inutiles n'ont pas été supprimées."

    # Vérifier que les colonnes utiles restent
    assert "id" in fe.data.columns, "La colonne 'id' a été supprimée à tort."


def test_log_transform_minutes(mock_data):
    """
    Teste la méthode `log_transform_minutes`.
    """
    fe = FeatEngineering(mock_data)
    fe.log_transform_minutes()

    # Vérifier que la colonne 'minutes' est transformée en 'log_minutes'
    assert "log_minutes" in fe.data.columns, "La colonne 'log_minutes' n'a pas été créée."
    assert "minutes" not in fe.data.columns, "La colonne 'minutes' n'a pas été supprimée."

    # Vérifier que les valeurs sont correctement transformées
    expected_log_values = np.log([10, 30])
    np.testing.assert_almost_equal(fe.data["log_minutes"].iloc[:2].values, expected_log_values, decimal=5)

    # Vérifier que 0 dans 'minutes' n'est pas transformé (logarithme indéfini)
    assert np.isneginf(fe.data["log_minutes"].iloc[2]), "Le logarithme de 0 n'est pas correctement géré."


def test_log_transform_minutes_missing_column(mock_data):
    """
    Teste le comportement de `log_transform_minutes` lorsque la colonne 'minutes' est absente.
    """
    mock_data.drop(columns=["minutes"], inplace=True)
    fe = FeatEngineering(mock_data)
    fe.log_transform_minutes()

    # Vérifier que la méthode ne génère pas d'erreur
    assert "log_minutes" not in fe.data.columns, "Une colonne 'log_minutes' a été créée alors que 'minutes' est absente."


def test_get_preprocessed_data(mock_data):
    """
    Teste la méthode `get_preprocessed_data`.
    """
    fe = FeatEngineering(mock_data)
    processed_data = fe.get_preprocessed_data()

    # Vérifier que la méthode retourne un DataFrame
    assert isinstance(processed_data, pd.DataFrame), "La méthode `get_preprocessed_data` ne retourne pas un DataFrame."

    # Vérifier que les dimensions sont correctes
    assert processed_data.shape == mock_data.shape, "Les dimensions du DataFrame retourné sont incorrectes."
