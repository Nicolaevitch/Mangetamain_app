import logging
from typing import Union
import pandas as pd
import numpy as np
from src.FindingCloseRecipes.config import NUMERIC_FEATURES, DEFAULT_WEIGHTS, COMBINED_WEIGHTS, TOP_N
from src.FindingCloseRecipes.distances import DistanceCalculator
from src.FindingCloseRecipes.vectorizers import Vectorizer
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

class RecipeFinder:
    """
    Une classe pour trouver des recettes similaires basées sur des distances pondérées.
    """

    def __init__(self, recipes_df: pd.DataFrame):
        """
        Initialise le RecipeFinder avec le DataFrame des recettes.

        Args:
            recipes_df (pd.DataFrame): DataFrame contenant les recettes avec leurs caractéristiques.
        """
        self.recipes_df = recipes_df
        self.id_to_index = pd.Series(recipes_df.index, index=recipes_df['id'])
        logger.info("RecipeFinder initialisé avec un DataFrame contenant %d recettes.", len(recipes_df))

    def preprocess(self) -> None:
        """
        Pré-traite les données nécessaires pour calculer les distances,
        notamment en créant les matrices creuses et en préparant les poids.
        """
        try:
            logger.info("Début du pré-traitement des données.")
            # Sélectionner les colonnes numériques
            self.numeric_df = self.recipes_df[NUMERIC_FEATURES]
            self.weights_array = np.array([DEFAULT_WEIGHTS[feature] for feature in NUMERIC_FEATURES])
            
            # Vectorisation des colonnes textuelles
            self.tfidf_name, _ = Vectorizer.tfidf_vectorize(self.recipes_df['name'])
            self.tfidf_tags, _ = Vectorizer.tfidf_vectorize(self.recipes_df['tags'])
            self.tfidf_steps, _ = Vectorizer.tfidf_vectorize(self.recipes_df['steps'])
            self.bow_ingredients, _ = Vectorizer.bow_vectorize(self.recipes_df['ingredients'])
            logger.info("Pré-traitement terminé avec succès.")
        except Exception as e:
            logger.error("Erreur pendant le pré-traitement : %s", str(e))
            raise

    def find_similar_recipes(self, recipe_id: Union[int, str]) -> pd.DataFrame:
        """
        Trouve les recettes similaires à une recette donnée, en utilisant
        des distances pondérées combinées.

        Args:
            recipe_id (Union[int, str]): L'identifiant de la recette cible.

        Returns:
            pd.DataFrame: DataFrame contenant les recettes similaires et leurs distances combinées.

        Raises:
            ValueError: Si l'identifiant de recette est introuvable.
        """
        try:
            if recipe_id not in self.id_to_index:
                logger.error("Identifiant de recette introuvable : %s", recipe_id)
                raise ValueError("Identifiant de recette introuvable.")
            
            logger.info("Recherche des recettes similaires pour la recette ID %s.", recipe_id)
            recipe_index = self.id_to_index[recipe_id]
            
            # Calcul des distances
            distance_name = DistanceCalculator.cosine_distance_sparse(
                recipe_id=recipe_id,
                tfidf_matrix=self.tfidf_name,
                id_to_index=self.id_to_index,
                index_to_id=self.id_to_index.index
            )
            distance_tags = DistanceCalculator.cosine_distance_sparse(
                recipe_id=recipe_id,
                tfidf_matrix=self.tfidf_tags,
                id_to_index=self.id_to_index,
                index_to_id=self.id_to_index.index
            )
            distance_steps = DistanceCalculator.cosine_distance_sparse(
                recipe_id=recipe_id,
                tfidf_matrix=self.tfidf_steps,
                id_to_index=self.id_to_index,
                index_to_id=self.id_to_index.index
            )
            distance_ingredients = DistanceCalculator.cosine_distance_sparse(
                recipe_id=recipe_id,
                tfidf_matrix=self.bow_ingredients,
                id_to_index=self.id_to_index,
                index_to_id=self.id_to_index.index
            )
            distance_numeric = DistanceCalculator.euclidean_distance(
                self.numeric_df, recipe_index, self.weights_array
            )

            # Combiner les distances avec les poids
            combined_distance = (
                COMBINED_WEIGHTS["alpha"] * distance_name +
                COMBINED_WEIGHTS["beta"] * distance_tags +
                COMBINED_WEIGHTS["gamma"] * distance_steps +
                COMBINED_WEIGHTS["delta"] * distance_ingredients +
                COMBINED_WEIGHTS["epsilon"] * distance_numeric
            )
            logger.debug("Distances combinées calculées avec succès.")

            # Trier par distances croissantes
            sorted_indices = np.argsort(combined_distance)
            sorted_indices = sorted_indices[sorted_indices != recipe_index]
            
            # Récupérer les top N recettes les plus proches
            top_n_indices = sorted_indices[:TOP_N]
            similar_recipes = self.recipes_df.iloc[top_n_indices].copy()
            similar_recipes['combined_distance'] = combined_distance[top_n_indices]

            logger.info("Recherche des recettes similaires terminée.")
            return similar_recipes
        except Exception as e:
            logger.error("Erreur lors de la recherche des recettes similaires : %s", str(e))
            raise




