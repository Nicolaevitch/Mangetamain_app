import os
import logging
import pandas as pd
from typing import List
from src.FindingCloseRecipes.recipe_finder import RecipeFinder

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

def reconstruct_pp_recipes() -> pd.DataFrame:
    """
    Reconstruit le DataFrame `pp_recipes` en combinant des fichiers CSV fragmentés.

    Returns:
        pd.DataFrame: Le DataFrame consolidé contenant les données de recettes.

    Raises:
        FileNotFoundError: Si aucun fichier n'est trouvé pour une colonne donnée.
    """
    datasets = {}
    columns = ["tags", "name", "steps", "ingredients", "numerics"]

    try:
        logger.info("Début de la reconstruction du DataFrame `pp_recipes`.")
        for column in columns:
            datasets[column] = []
            for i in range(1, 5):
                file_path = f"data/pp_recipes_{column}_{i}.csv"
                if os.path.exists(file_path):
                    datasets[column].append(pd.read_csv(file_path))
                    logger.debug("Chargement du fichier : %s", file_path)
                else:
                    logger.warning("Fichier manquant : %s", file_path)

        # Concaténation des fichiers pour chaque colonne
        tags = pd.concat(datasets["tags"], ignore_index=True) if datasets["tags"] else pd.DataFrame()
        name = pd.concat(datasets["name"], ignore_index=True) if datasets["name"] else pd.DataFrame()
        steps = pd.concat(datasets["steps"], ignore_index=True) if datasets["steps"] else pd.DataFrame()
        ingredients = pd.concat(datasets["ingredients"], ignore_index=True) if datasets["ingredients"] else pd.DataFrame()
        numerics = pd.concat(datasets["numerics"], ignore_index=True) if datasets["numerics"] else pd.DataFrame()

        # Fusion des datasets selon la colonne "id"
        if not tags.empty:
            pp_recipes = tags
        else:
            raise FileNotFoundError("Les fichiers pour la colonne 'tags' sont introuvables.")

        for df in [name, steps, ingredients, numerics]:
            if not df.empty:
                pp_recipes = pp_recipes.merge(df, on="id", how="inner")
                logger.debug("Fusion réussie avec un DataFrame supplémentaire.")

        logger.info("Reconstruction du DataFrame `pp_recipes` terminée avec succès.")
        return pp_recipes
    except Exception as e:
        logger.error("Erreur lors de la reconstruction du DataFrame : %s", str(e))
        raise

def run_recipe_finder(recipe_id: int) -> pd.DataFrame:
    """
    Trouve les 100 recettes les plus proches d'une recette donnée par son ID.

    Args:
        recipe_id (int): L'identifiant de la recette pour laquelle chercher les recettes similaires.

    Returns:
        pd.DataFrame: Les 100 recettes les plus proches avec leurs distances combinées.

    Raises:
        ValueError: Si l'ID de la recette est introuvable dans le DataFrame.
    """
    try:
        logger.info("Début de la recherche pour la recette ID %d.", recipe_id)

        # Charger le dataset
        pp_recipes = reconstruct_pp_recipes()

        # Initialiser le RecipeFinder
        finder = RecipeFinder(pp_recipes)
        finder.preprocess()  # Cette étape inclut la préparation des matrices creuses

        # Trouver les recettes similaires
        similar_recipes = finder.find_similar_recipes(recipe_id)

        # Affichage pour vérification
        logger.info("Recherche des recettes similaires terminée pour l'ID %d.", recipe_id)
        logger.debug("Recette cible : \n%s", pp_recipes[pp_recipes['id'] == recipe_id])
        logger.debug("Recettes similaires trouvées : \n%s", similar_recipes)

        return similar_recipes
    except ValueError as e:
        logger.error("Erreur : %s", str(e))
        raise
    except Exception as e:
        logger.critical("Une erreur critique s'est produite : %s", str(e))
        raise




