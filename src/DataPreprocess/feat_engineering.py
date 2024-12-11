import logging
import pandas as pd
import numpy as np
from typing import List

# Assurez-vous que le dossier logs existe
import os
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

class FeatEngineering:
    """
    Classe pour effectuer le feature engineering sur un DataFrame contenant des données de recettes.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialise une instance de FeatEngineering.

        Args:
            data (pd.DataFrame): DataFrame contenant les données brutes.

        Raises:
            TypeError: Si l'entrée n'est pas un DataFrame Pandas.
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Le paramètre 'data' doit être un DataFrame Pandas.")
            raise TypeError("Le paramètre 'data' doit être un DataFrame Pandas.")
        
        self.data = data.copy()
        logger.info("Initialisation de FeatEngineering avec un DataFrame de dimensions %s.", self.data.shape)

    def extract_nutrition_features(self) -> "FeatEngineering":
        """
        Sépare la colonne 'nutrition' en plusieurs colonnes pour chaque élément nutritionnel
        et nettoie les valeurs des colonnes 'calories' et 'carbohydrates (PDV%)'.

        Returns:
            FeatEngineering: L'instance mise à jour de FeatEngineering.

        Raises:
            KeyError: Si la colonne 'nutrition' est absente.
            ValueError: Si les données de 'nutrition' ne peuvent pas être transformées correctement.
        """
        if 'nutrition' not in self.data.columns:
            logger.error("La colonne 'nutrition' est absente du DataFrame.")
            raise KeyError("La colonne 'nutrition' est absente du DataFrame.")
        
        try:
            logger.info("Extraction des caractéristiques nutritionnelles depuis la colonne 'nutrition'.")
            nutrition_columns = [
                'calories', 'total fat (PDV%)', 'sugar (PDV%)', 'sodium (PDV%)',
                'protein (PDV%)', 'saturated fat (PDV%)', 'carbohydrates (PDV%)'
            ]
            self.data[nutrition_columns] = self.data['nutrition'].str.split(",", expand=True)

            # Nettoyage des données
            self.data['calories'] = self.data['calories'].str.replace('[', '', regex=False)
            self.data['carbohydrates (PDV%)'] = self.data['carbohydrates (PDV%)'].str.replace(']', '', regex=False)

            # Conversion en float
            self.data[nutrition_columns] = self.data[nutrition_columns].astype(float)
            logger.info("Extraction et nettoyage des caractéristiques nutritionnelles terminés.")
        except Exception as e:
            logger.error("Erreur lors de l'extraction des caractéristiques nutritionnelles : %s", str(e))
            raise ValueError("Impossible de traiter les données nutritionnelles.")
        
        return self

    def drop_useless_features(self) -> "FeatEngineering":
        """
        Supprime les colonnes inutiles pour l'analyse.

        Returns:
            FeatEngineering: L'instance mise à jour de FeatEngineering.
        """
        columns_to_drop = ['submitted', 'nutrition', 'description', 'n_steps', 'n_ingredients']
        initial_columns = set(self.data.columns)
        self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        removed_columns = initial_columns - set(self.data.columns)
        logger.info("Colonnes supprimées : %s", ", ".join(removed_columns) if removed_columns else "Aucune colonne supprimée.")
        return self

    def log_transform_minutes(self) -> "FeatEngineering":
        """
        Applique une transformation logarithmique à la colonne 'minutes' et la renomme en 'log_minutes'.

        Returns:
            FeatEngineering: L'instance mise à jour de FeatEngineering.

        Raises:
            KeyError: Si la colonne 'minutes' est absente.
        """
        if 'minutes' not in self.data.columns:
            logger.warning("La colonne 'minutes' est absente. Transformation logarithmique ignorée.")
        else:
            logger.info("Application de la transformation logarithmique sur la colonne 'minutes'.")
            self.data['log_minutes'] = np.log(self.data['minutes'])
            self.data.drop(columns=['minutes'], inplace=True)
            logger.info("Transformation logarithmique terminée avec succès.")
        return self

    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Retourne le DataFrame après toutes les transformations.

        Returns:
            pd.DataFrame: Le DataFrame prétraité.
        """
        logger.info("Renvoi du DataFrame prétraité de dimensions %s.", self.data.shape)
        return self.data

