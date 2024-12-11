import pytest
import pandas as pd
import os
from src.DataPreprocess.data_cleaning import DataCleaning


@pytest.fixture
def sample_data():
    """
    Fixture pour créer un DataFrame simulé pour les tests.
    """
    data = {
        "id": [1, 2, 3, 4, 5],
        "minutes": [10, 0, 43201, 20, 5],
        "calories": [500, 10000, 15000, 300, 400],
        "ingredients": [
            "['salt', 'pepper']",
            "['sugar', 'flour']",
            "['butter', 'milk']",
            "['water', 'lemon']",
            "['honey', 'tea']"
        ],
        "other_column": ["test1", "test2", "test3", "test4", "test5"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def ingredient_map_file(tmpdir):
    """
    Fixture pour créer un fichier CSV simulé pour le mapping des ingrédients.
    """
    file_path = tmpdir.join("ingredient_map.csv")
    mapping_data = {"raw_ingr": ["salt", "sugar", "butter", "water", "honey"],
                    "replaced": ["spice", "sweetener", "dairy", "liquid", "sweetener"]}
    pd.DataFrame(mapping_data).to_csv(file_path, index=False)
    return str(file_path)


def test_remove_long_recipes(sample_data):
    """
    Teste la méthode `remove_long_recipes`.
    """
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.remove_long_recipes(max_minutes=24 * 60).get_cleaned_data()

    # Vérifier que les recettes avec plus de 1440 minutes sont supprimées
    assert cleaned_data["minutes"].max() <= 1440, "Les recettes longues n'ont pas été supprimées."


def test_replace_zero_minutes(sample_data):
    """
    Teste la méthode `replace_zero_minutes`.
    """
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.replace_zero_minutes(replacement_minutes=8).get_cleaned_data()

    # Vérifier que les valeurs 0 dans 'minutes' ont été remplacées
    assert (cleaned_data["minutes"] == 0).sum() == 0, "Les recettes avec 0 minutes n'ont pas été remplacées."
    assert (cleaned_data["minutes"] == 8).sum() > 0, "La valeur 8 n'a pas été correctement remplacée."


def test_remove_high_calories_recipes(sample_data):
    """
    Teste la méthode `remove_high_calories_recipes`.
    """
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.remove_high_calories_recipes(max_calories=10000).get_cleaned_data()

    # Vérifier que les recettes avec des calories > 10000 sont supprimées
    assert cleaned_data["calories"].max() <= 10000, "Les recettes avec des calories élevées n'ont pas été supprimées."


def test_map_ingredients(sample_data, ingredient_map_file):
    """
    Teste la méthode `map_ingredients`.
    """
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.map_ingredients(ingredient_map_file).get_cleaned_data()

    # Vérifier que les ingrédients ont été correctement remplacés
    assert "spice" in cleaned_data["ingredients"].iloc[0], "Le mapping des ingrédients n'a pas été appliqué."
    assert "sweetener" in cleaned_data["ingredients"].iloc[1], "Le mapping des ingrédients n'a pas été appliqué."


def test_map_ingredients_missing_file(sample_data):
    """
    Teste la méthode `map_ingredients` avec un fichier de mapping manquant.
    """
    cleaner = DataCleaning(sample_data)
    with pytest.raises(FileNotFoundError):
        cleaner.map_ingredients("missing_file.csv")


def test_handle_missing_values(sample_data):
    """
    Teste la méthode `handle_missing_values`.
    """
    sample_data.loc[0, "other_column"] = None  # Ajouter une valeur manquante
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.handle_missing_values().get_cleaned_data()

    # Vérifier que les lignes avec des NaN ont été supprimées
    assert cleaned_data.isna().sum().sum() == 0, "Les valeurs manquantes n'ont pas été supprimées."
    assert len(cleaned_data) < len(sample_data), "Aucune ligne n'a été supprimée malgré des valeurs manquantes."


def test_get_cleaned_data(sample_data):
    """
    Teste la méthode `get_cleaned_data`.
    """
    cleaner = DataCleaning(sample_data)
    cleaned_data = cleaner.get_cleaned_data()

    # Vérifier que le DataFrame nettoyé est retourné
    assert isinstance(cleaned_data, pd.DataFrame), "Le résultat retourné n'est pas un DataFrame."
    assert not cleaned_data.empty, "Le DataFrame nettoyé est vide."
