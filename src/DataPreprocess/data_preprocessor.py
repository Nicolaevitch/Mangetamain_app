import os
import logging
import pandas as pd
from typing import List, Optional
from src.DataPreprocess.normalizer import Normalizer
from src.DataPreprocess.feat_engineering import FeatEngineering
from src.DataPreprocess.data_cleaning import DataCleaning
from src.DataPreprocess.vectorizer_preparator import VectorizerPreparator
from src.DataPreprocess.split_dataset import DatasetSplitter

# Assurez-vous que le dossier logs existe
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log", mode="a"),
        logging.FileHandler("logs/error.log", mode="a"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Classe pour charger, nettoyer, traiter et sauvegarder les données pour une analyse approfondie.
    """

    def __init__(self, file_path: str, ingredient_map_path: str):
        """
        Initialise une instance de DataPreprocessor.

        Args:
            file_path (str): Chemin vers le fichier de données brut.
            ingredient_map_path (str): Chemin vers le fichier de mapping des ingrédients.

        Raises:
            FileNotFoundError: Si le fichier de données ou le fichier de mapping n'existe pas.
        """
        if not os.path.exists(file_path):
            logger.error("Fichier de données introuvable : %s", file_path)
            raise FileNotFoundError(f"Fichier de données introuvable : {file_path}")
        if not os.path.exists(ingredient_map_path):
            logger.error("Fichier de mapping des ingrédients introuvable : %s", ingredient_map_path)
            raise FileNotFoundError(f"Fichier de mapping des ingrédients introuvable : {ingredient_map_path}")

        self.file_path = file_path
        self.ingredient_map_path = ingredient_map_path
        self.data: Optional[pd.DataFrame] = None
        logger.info("DataPreprocessor initialisé avec les fichiers : %s, %s", file_path, ingredient_map_path)

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis le fichier CSV.

        Returns:
            pd.DataFrame: Les données chargées.
        """
        logger.info("Chargement des données depuis %s", self.file_path)
        self.data = pd.read_csv(self.file_path)
        logger.info("Données chargées avec succès : %s lignes, %s colonnes", self.data.shape[0], self.data.shape[1])
        return self.data

    def save_data(self, output_path: str) -> None:
        """
        Sauvegarde les données prétraitées dans un fichier CSV.

        Args:
            output_path (str): Chemin vers le fichier de sortie.
        """
        logger.info("Sauvegarde des données prétraitées dans %s", output_path)
        self.data.to_csv(output_path, index=False)
        logger.info("Données sauvegardées avec succès.")

    def preprocess(self) -> pd.DataFrame:
        """
        Pipeline complet de prétraitement des données.

        Returns:
            pd.DataFrame: Les données prétraitées.
        """
        if self.data is None:
            logger.error("Aucune donnée à prétraiter. Appelez 'load_data' avant.")
            raise ValueError("Aucune donnée chargée. Appelez 'load_data' avant d'exécuter le prétraitement.")

        try:
            logger.info("Début du pipeline de prétraitement.")
            
            # Étape 1 : Nettoyage des données
            cleaner = DataCleaning(self.data)
            self.data = (
                cleaner
                .replace_zero_minutes(replacement_minutes=8)
                .remove_long_recipes(max_minutes=30 * 24 * 60)
                .map_ingredients(self.ingredient_map_path)
                .get_cleaned_data()
            )

            # Étape 2 : Feature Engineering
            feat_engineer = FeatEngineering(self.data)
            self.data = (
                feat_engineer
                .extract_nutrition_features()
                .drop_useless_features()
                .log_transform_minutes()
                .get_preprocessed_data()
            )

            # Supprimer les lignes avec des NaN après le feature engineering
            cleaner = DataCleaning(self.data)
            self.data = cleaner.handle_missing_values().get_cleaned_data()

            # Étape 3 : Suppression des recettes riches en calories
            self.data = (
                DataCleaning(self.data)
                .remove_high_calories_recipes(max_calories=10000)
                .get_cleaned_data()
            )

            # Étape 4 : Préparation pour la vectorisation
            vectorizer = VectorizerPreparator(self.data)
            self.data = (
                vectorizer
                .process_ingredients()
                .process_steps()
                .process_name()
                .process_tags()
                .get_prepared_data()
            )

            # Étape 5 : Normalisation
            normalizer = Normalizer()
            self.data = normalizer.normalize(self.data)

            # Supprimer les lignes avec des NaN après normalisation
            cleaner = DataCleaning(self.data)
            self.data = cleaner.handle_missing_values().get_cleaned_data()

            # Étape 6 : Séparation des colonnes en datasets
            splitter = DatasetSplitter(self.data, output_dir="data/split_datasets")
            splitter.split_by_column(["tags", "steps", "ingredients", "name"])

            numeric_columns = [
                "log_minutes", "calories", "total fat (PDV%)", "sugar (PDV%)",
                "sodium (PDV%)", "protein (PDV%)", "saturated fat (PDV%)", "carbohydrates (PDV%)"
            ]
            splitter.split_by_numeric_columns(numeric_columns)

            # Diviser chaque dataset en 4 parties
            datasets = [
                "data/split_datasets/pp_recipes_tags.csv",
                "data/split_datasets/pp_recipes_steps.csv",
                "data/split_datasets/pp_recipes_ingredients.csv",
                "data/split_datasets/pp_recipes_name.csv",
                "data/split_datasets/pp_recipes_numerics.csv",
            ]
            for dataset_file in datasets:
                splitter.split_into_parts(input_file=dataset_file, num_parts=4)

            logger.info("Pipeline de prétraitement terminé avec succès.")
            return self.data

        except Exception as e:
            logger.error("Erreur lors du pipeline de prétraitement : %s", str(e))
            raise






