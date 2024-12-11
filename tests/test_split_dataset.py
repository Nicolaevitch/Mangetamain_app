import pytest
import pandas as pd
import os
from unittest.mock import patch
from src.DataPreprocess.split_dataset import DatasetSplitter


@pytest.fixture
def sample_data():
    """
    Fixture pour créer un DataFrame d'exemple.
    """
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Recipe A", "Recipe B", "Recipe C", "Recipe D", "Recipe E"],
        "tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],
        "steps": ["step1", "step2", "step3", "step4", "step5"],
        "minutes": [10, 20, 30, 40, 50],
        "calories": [100, 200, 300, 400, 500],
    }
    return pd.DataFrame(data)


@pytest.fixture
def output_dir(tmpdir):
    """
    Fixture pour créer un dossier temporaire.
    """
    return str(tmpdir)


def test_split_by_column(sample_data, output_dir):
    """
    Teste la méthode `split_by_column` pour vérifier la sauvegarde des colonnes spécifiées.
    """
    splitter = DatasetSplitter(sample_data, output_dir)

    # Exécuter la division
    splitter.split_by_column(["name", "tags"])

    # Vérifier les fichiers générés
    for column in ["name", "tags"]:
        output_file = os.path.join(output_dir, f"pp_recipes_{column}.csv")
        assert os.path.exists(output_file), f"Le fichier pour la colonne '{column}' n'a pas été généré."
        df = pd.read_csv(output_file)
        assert "id" in df.columns and column in df.columns, f"Les colonnes attendues ne sont pas présentes dans '{output_file}'."


def test_split_by_column_invalid_column(sample_data, output_dir):
    """
    Teste que `split_by_column` lève une exception si une colonne est manquante.
    """
    splitter = DatasetSplitter(sample_data, output_dir)

    with pytest.raises(ValueError, match="n'existe pas dans le DataFrame"):
        splitter.split_by_column(["invalid_column"])


def test_split_by_numeric_columns(sample_data, output_dir):
    """
    Teste la méthode `split_by_numeric_columns` pour vérifier la sauvegarde des colonnes numériques spécifiées.
    """
    splitter = DatasetSplitter(sample_data, output_dir)

    # Exécuter la division
    splitter.split_by_numeric_columns(["minutes", "calories"])

    # Vérifier le fichier généré
    output_file = os.path.join(output_dir, "pp_recipes_numerics.csv")
    assert os.path.exists(output_file), "Le fichier pour les colonnes numériques n'a pas été généré."
    df = pd.read_csv(output_file)
    assert "id" in df.columns and "minutes" in df.columns and "calories" in df.columns, \
        "Les colonnes attendues ne sont pas présentes dans le fichier."


def test_split_by_numeric_columns_invalid_column(sample_data, output_dir):
    """
    Teste que `split_by_numeric_columns` lève une exception si une colonne est manquante.
    """
    splitter = DatasetSplitter(sample_data, output_dir)

    with pytest.raises(ValueError, match="n'existe pas dans le DataFrame"):
        splitter.split_by_numeric_columns(["invalid_column"])


def test_split_into_parts(sample_data, output_dir):
    """
    Teste la méthode `split_into_parts` pour vérifier la division d'un fichier CSV en parties.
    """
    # Sauvegarder un fichier CSV pour le test
    input_file = os.path.join(output_dir, "test_file.csv")
    sample_data.to_csv(input_file, index=False)

    splitter = DatasetSplitter(sample_data, output_dir)

    # Exécuter la division
    splitter.split_into_parts(input_file, num_parts=3)

    # Vérifier les fichiers générés
    for i in range(1, 4):
        output_file = os.path.join(output_dir, f"test_file_part_{i}.csv")
        assert os.path.exists(output_file), f"Le fichier '{output_file}' n'a pas été généré."

    # Vérifier le nombre total de lignes
    total_rows = sum(pd.read_csv(os.path.join(output_dir, f"test_file_part_{i}.csv")).shape[0] for i in range(1, 4))
    assert total_rows == len(sample_data), "Le nombre total de lignes dans les parties ne correspond pas au fichier original."


def test_split_into_parts_invalid_file(output_dir):
    """
    Teste que `split_into_parts` lève une exception si le fichier n'existe pas.
    """
    splitter = DatasetSplitter(pd.DataFrame(), output_dir)

    with pytest.raises(FileNotFoundError, match="n'existe pas"):
        splitter.split_into_parts("invalid_file.csv", num_parts=3)


def test_split_into_parts_invalid_num_parts(sample_data, output_dir):
    """
    Teste que `split_into_parts` lève une exception si le nombre de parties est invalide.
    """
    splitter = DatasetSplitter(sample_data, output_dir)

    with pytest.raises(ValueError, match="doit être au moins égal à 1"):
        splitter.split_into_parts("test_file.csv", num_parts=0)
