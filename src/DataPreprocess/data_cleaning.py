import logging
import pandas as pd
import ast
from typing import Union

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

class DataCleaning:
    """
    Classe pour effectuer le nettoyage des données, en supprimant les outliers,
    en traitant les anomalies et en manipulant les ingrédients.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'instance de DataCleaning.

        Args:
            data (pd.DataFrame): DataFrame contenant les données brutes.
        """
        if not isinstance(data, pd.DataFrame):
            logger.error("Le paramètre 'data' doit être un DataFrame Pandas.")
            raise TypeError("Le paramètre 'data' doit être un DataFrame Pandas.")
        
        self.data = data.copy()
        logger.info("Initialisation de la classe DataCleaning avec un DataFrame de dimensions %s.", self.data.shape)

    def remove_long_recipes(self, max_minutes: int = 30 * 24 * 60) -> "DataCleaning":
        """
        Supprime les recettes dont le temps de préparation dépasse une limite.

        Args:
            max_minutes (int): Temps maximal en minutes (par défaut : 30 jours).

        Returns:
            DataCleaning: L'instance mise à jour de DataCleaning.
        """
        initial_count = len(self.data)
        self.data = self.data[self.data['minutes'] <= max_minutes]
        removed_count = initial_count - len(self.data)
        logger.info("Suppression des recettes de plus de %d minutes : %d recettes supprimées.", max_minutes, removed_count)
        return self

    def replace_zero_minutes(self, replacement_minutes: int = 8) -> "DataCleaning":
        """
        Remplace les temps de préparation égaux à 0 par une valeur par défaut.

        Args:
            replacement_minutes (int): Valeur de remplacement (par défaut : 8 minutes).

        Returns:
            DataCleaning: L'instance mise à jour de DataCleaning.
        """
        zero_count = (self.data['minutes'] == 0).sum()
        self.data.loc[self.data['minutes'] == 0, 'minutes'] = replacement_minutes
        logger.info("Remplacement de %d recettes avec 0 minutes par %d minutes.", zero_count, replacement_minutes)
        return self

    def remove_high_calories_recipes(self, max_calories: int = 10000) -> "DataCleaning":
        """
        Supprime les recettes ayant des calories au-dessus d'une valeur seuil.

        Args:
            max_calories (int): Seuil maximal de calories (par défaut : 10,000).

        Returns:
            DataCleaning: L'instance mise à jour de DataCleaning.
        """
        if 'calories' in self.data.columns:
            initial_count = len(self.data)
            self.data = self.data[self.data['calories'] <= max_calories]
            removed_count = initial_count - len(self.data)
            logger.info("Suppression des recettes avec plus de %d calories : %d recettes supprimées.", max_calories, removed_count)
        else:
            logger.warning("La colonne 'calories' est absente du DataFrame.")
        return self

    def map_ingredients(self, ingredient_map_path: str) -> "DataCleaning":
        """
        Remplace les noms d'ingrédients par des catégories générales à l'aide d'un fichier de mapping.

        Args:
            ingredient_map_path (str): Chemin vers le fichier CSV contenant les mappings.

        Returns:
            DataCleaning: L'instance mise à jour de DataCleaning.

        Raises:
            FileNotFoundError: Si le fichier de mapping est introuvable.
        """
        if not os.path.exists(ingredient_map_path):
            logger.error("Fichier de mapping des ingrédients introuvable : %s.", ingredient_map_path)
            raise FileNotFoundError(f"Fichier de mapping introuvable : {ingredient_map_path}")
        
        logger.info("Chargement du fichier de mapping des ingrédients : %s.", ingredient_map_path)
        ingr_map = pd.read_csv(ingredient_map_path)
        ingredient_mapping = dict(zip(ingr_map['raw_ingr'], ingr_map['replaced']))

        def replace_ingredients(ingredient_list: str) -> List[str]:
            try:
                ingredients = ast.literal_eval(ingredient_list)
                return [ingredient_mapping.get(ingredient, ingredient) for ingredient in ingredients]
            except (ValueError, SyntaxError):
                logger.error("Erreur lors du traitement des ingrédients : %s", ingredient_list)
                return ingredient_list  # Retourner la liste brute en cas d'erreur

        if 'ingredients' in self.data.columns:
            self.data['ingredients'] = self.data['ingredients'].apply(replace_ingredients)
            logger.info("Mapping des ingrédients terminé avec succès.")
        else:
            logger.warning("La colonne 'ingredients' est absente du DataFrame.")
        return self

    def handle_missing_values(self) -> "DataCleaning":
        """
        Gère les valeurs manquantes en remplaçant les chaînes vides et 'None' par NaN,
        et en supprimant les lignes contenant des NaN.

        Returns:
            DataCleaning: L'instance mise à jour de DataCleaning.
        """
        initial_count = len(self.data)
        self.data.replace(["", "None"], pd.NA, inplace=True)
        self.data.dropna(inplace=True)
        removed_count = initial_count - len(self.data)
        logger.info("Gestion des valeurs manquantes : %d lignes supprimées.", removed_count)
        return self

    def get_cleaned_data(self) -> pd.DataFrame:
        """
        Retourne le DataFrame nettoyé.

        Returns:
            pd.DataFrame: Le DataFrame nettoyé.
        """
        logger.info("Renvoi du DataFrame nettoyé de dimensions %s.", self.data.shape)
        return self.data

