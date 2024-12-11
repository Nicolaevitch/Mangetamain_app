import pytest
import pandas as pd
from src.DataPreprocess.vectorizer_preparator import VectorizerPreparator


@pytest.fixture
def sample_data():
    """
    Fixture pour créer un DataFrame d'exemple avec des colonnes textuelles.
    """
    data = {
        "id": [1, 2, 3],
        "ingredients": ['["sugar", "milk", "flour"]', '["butter", "egg"]', '["salt", "water"]'],
        "steps": ['["Mix ingredients", "Bake"]', '["Whisk butter", "Add egg"]', '["Boil water", "Add salt"]'],
        "name": ["Cake Recipe", "Pancake Recipe", "Soup Recipe"],
        "tags": ['["dessert", "easy"]', '["breakfast", "quick"]', '["dinner", "healthy"]']
    }
    return pd.DataFrame(data)


def test_process_ingredients(sample_data):
    """
    Teste le traitement de la colonne 'ingredients'.
    """
    preparator = VectorizerPreparator(sample_data)
    preparator.process_ingredients()
    result = preparator.get_prepared_data()

    assert "ingredients" in result.columns, "La colonne 'ingredients' devrait être présente."
    assert result.loc[0, "ingredients"] == "sugar milk flour", "Les ingrédients ne sont pas correctement transformés."


def test_process_steps(sample_data):
    """
    Teste le traitement de la colonne 'steps'.
    """
    preparator = VectorizerPreparator(sample_data)
    preparator.process_steps()
    result = preparator.get_prepared_data()

    assert "steps" in result.columns, "La colonne 'steps' devrait être présente."
    assert "mix" in result.loc[0, "steps"], "Les étapes ne sont pas correctement traitées (ex. stemming ou stop words)."


def test_process_name(sample_data):
    """
    Teste le traitement de la colonne 'name'.
    """
    preparator = VectorizerPreparator(sample_data)
    preparator.process_name()
    result = preparator.get_prepared_data()

    assert "name" in result.columns, "La colonne 'name' devrait être présente."
    assert "cake" in result.loc[0, "name"].lower(), "Les noms ne sont pas correctement traités (ex. stemming ou stop words)."


def test_process_tags(sample_data):
    """
    Teste le traitement de la colonne 'tags'.
    """
    preparator = VectorizerPreparator(sample_data)
    preparator.process_tags()
    result = preparator.get_prepared_data()

    assert "tags" in result.columns, "La colonne 'tags' devrait être présente."
    assert result.loc[0, "tags"] == "dessert easy", "Les tags ne sont pas correctement transformés."


def test_missing_columns(sample_data):
    """
    Teste que le traitement gère correctement l'absence de colonnes.
    """
    preparator = VectorizerPreparator(sample_data.drop(columns=["steps", "name"]))
    preparator.process_steps()
    preparator.process_name()

    result = preparator.get_prepared_data()

    assert "steps" not in result.columns, "La colonne 'steps' ne devrait pas être présente."
    assert "name" not in result.columns, "La colonne 'name' ne devrait pas être présente."


def test_get_prepared_data(sample_data):
    """
    Teste la méthode `get_prepared_data`.
    """
    preparator = VectorizerPreparator(sample_data)
    preparator.process_ingredients().process_steps().process_name().process_tags()
    result = preparator.get_prepared_data()

    assert isinstance(result, pd.DataFrame), "Le résultat doit être un DataFrame."
    assert result.shape == sample_data.shape, "Le DataFrame transformé devrait avoir les mêmes dimensions que l'original."
