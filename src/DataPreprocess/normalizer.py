import logging
from typing import List
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

class Normalizer:
    """
    Classe pour normaliser les colonnes numériques d'un DataFrame à l'aide de StandardScaler.
    """

    def __init__(self):
        """
        Initialise l'instance de Normalizer avec un StandardScaler.
        """
        self.scaler = StandardScaler()

    def normalize(self, data: pd.DataFrame, columns_to_normalize: List[str]) -> pd.DataFrame:
        """
        Normalise les colonnes numériques spécifiées dans un DataFrame.

        Args:
            data (pd.DataFrame): Le DataFrame contenant les colonnes à normaliser.
            columns_to_normalize (List[str]): Liste des colonnes à normaliser.

        Returns:
            pd.DataFrame: Le DataFrame avec les colonnes spécifiées normalisées.

        Raises:
            ValueError: Si une ou plusieurs colonnes spécifiées ne sont pas présentes dans le DataFrame.
        """
        # Vérification de la présence des colonnes à normaliser
        missing_columns = [col for col in columns_to_normalize if col not in data.columns]
        if missing_columns:
            logger.error("Les colonnes suivantes sont absentes du DataFrame : %s", missing_columns)
            raise ValueError(f"Colonnes manquantes dans le DataFrame : {missing_columns}")

        try:
            logger.info("Début de la normalisation des colonnes : %s", columns_to_normalize)
            # Normalisation des colonnes
            data[columns_to_normalize] = self.scaler.fit_transform(data[columns_to_normalize])
            logger.info("Normalisation terminée avec succès.")
            return data
        except Exception as e:
            logger.error("Erreur lors de la normalisation : %s", str(e))
            raise
