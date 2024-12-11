import logging
from src.DataPreprocess.data_preprocessor import DataPreprocessor

# Configuration du logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log", mode="a"),
        logging.FileHandler("logs/error.log", mode="a"),
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Point d'entrée principal pour exécuter le pipeline de prétraitement des données.
    Charge les données brutes, applique le pipeline de nettoyage et de transformation,
    et sauvegarde les données prétraitées.

    Les fichiers requis :
    - `Raw_recipes.csv` : Dataset brut contenant les recettes.
    - `ingr_map.csv` : Mapping des ingrédients pour le nettoyage.
    """
    try:
        # Chemins des fichiers
        file_path = "data/Raw_recipes.csv"
        ingredient_map_path = "data/ingr_map.csv"
        output_path = "data/pp_recipes.csv"

        logger.info("Initialisation du prétraitement.")
        logger.info(f"Chargement des données depuis : {file_path}")

        # Initialiser le préprocesseur
        preprocessor = DataPreprocessor(file_path, ingredient_map_path)

        # Charger les données
        preprocessor.load_data()

        # Appliquer le pipeline de prétraitement
        preprocessor.preprocess()

        # Sauvegarder les données prétraitées
        preprocessor.save_data(output_path)

        logger.info(f"Prétraitement terminé. Données sauvegardées dans : {output_path}")
        print("Prétraitement terminé. Données sauvegardées dans :", output_path)
    except FileNotFoundError as e:
        logger.error(f"Fichier non trouvé : {e}")
        print(f"Erreur : {e}")
    except Exception as e:
        logger.critical(f"Erreur critique lors du prétraitement : {e}", exc_info=True)
        print(f"Une erreur critique s'est produite : {e}")

if __name__ == "__main__":
    main()
