import logging
from typing import Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import os

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

class DistanceCalculator:
    """
    Une classe utilitaire pour calculer les distances entre recettes
    basées sur des caractéristiques numériques ou des matrices creuses.
    """

    @staticmethod
    def euclidean_distance(numeric_df: pd.DataFrame, recipe_index: int, weights_array: np.ndarray) -> np.ndarray:
        """
        Calcule la distance euclidienne pondérée pour des données numériques.

        Args:
            numeric_df (pd.DataFrame): DataFrame contenant les caractéristiques numériques.
            recipe_index (int): L'index de la recette cible.
            weights_array (np.ndarray): Tableau des poids pour chaque caractéristique.

        Returns:
            np.ndarray: Tableau des distances pondérées calculées.
        """
        if recipe_index < 0 or recipe_index >= len(numeric_df):
            logger.error("Index de recette invalide pour la distance euclidienne.")
            raise ValueError("Index de recette invalide.")

        logger.info("Calcul des distances euclidiennes pondérées.")
        recipe_vector = numeric_df.iloc[recipe_index].values
        differences = numeric_df.values - recipe_vector
        squared_diff = differences ** 2
        weighted_squared_diff = squared_diff * weights_array
        distances = np.sqrt(np.sum(weighted_squared_diff, axis=1))
        logger.info("Calcul des distances euclidiennes terminé.")
        return distances

    @staticmethod
    def cosine_distance_sparse(
        recipe_id: Union[int, str],
        tfidf_matrix: csr_matrix,
        id_to_index: pd.Series,
        index_to_id: pd.Series
    ) -> np.ndarray:
        """
        Calcule la distance cosinus pour une matrice creuse (sparse).

        Args:
            recipe_id (Union[int, str]): L'identifiant de la recette cible.
            tfidf_matrix (csr_matrix): Matrice TF-IDF creuse (sparse).
            id_to_index (pd.Series): Mapping des identifiants de recette aux indices.
            index_to_id (pd.Series): Mapping des indices aux identifiants de recette.

        Returns:
            np.ndarray: Tableau des distances cosinus calculées.

        Raises:
            ValueError: Si l'identifiant de recette est introuvable.
        """
        if recipe_id not in id_to_index:
            logger.error(f"Identifiant de recette introuvable : {recipe_id}")
            raise ValueError("Identifiant de recette introuvable.")

        logger.info(f"Calcul des distances cosinus pour la recette ID {recipe_id}.")
        recipe_idx = id_to_index[recipe_id]

        # Extraire le vecteur TF-IDF de la recette cible
        recipe_vector = tfidf_matrix[recipe_idx]

        # Calculer les similarités cosinus
        cosine_similarities = cosine_similarity(recipe_vector, tfidf_matrix).flatten()

        # Calculer les distances
        distances = 1 - cosine_similarities
        logger.info(f"Calcul des distances cosinus terminé pour la recette ID {recipe_id}.")
        return distances

    
    

