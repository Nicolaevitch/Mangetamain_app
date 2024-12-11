import ast
import logging
from typing import Any
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pandas as pd

# Télécharger les ressources NLTK si nécessaire
nltk.download("stopwords")
nltk.download("punkt")

# Configurer le logger
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


class VectorizerPreparator:
    """
    Classe pour préparer les données textuelles à la vectorisation.
    Elle nettoie, transforme et optimise les colonnes textuelles d'un DataFrame.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialise l'instance du VectorizerPreparator.

        Args:
            data (pd.DataFrame): DataFrame contenant les colonnes textuelles à transformer.
        """
        self.data = data.copy()
        self.stemmer = SnowballStemmer("english")
        self.stop_words = set(stopwords.words("english"))
        logger.info("VectorizerPreparator initialisé avec succès.")

    def process_ingredients(self) -> "VectorizerPreparator":
        """
        Transforme la colonne 'ingredients' en une chaîne de caractères.

        Returns:
            VectorizerPreparator: L'instance actuelle après transformation.
        """
        if "ingredients" in self.data.columns:
            logger.info("Traitement de la colonne 'ingredients'.")
            self.data["ingredients"] = self.data["ingredients"].apply(lambda x: " ".join(ast.literal_eval(x)))
        else:
            logger.warning("La colonne 'ingredients' n'est pas présente dans le DataFrame.")
        return self

    def process_steps(self) -> "VectorizerPreparator":
        """
        Traite la colonne 'steps' en enlevant les stop words, la ponctuation, 
        et en appliquant le stemming.

        Returns:
            VectorizerPreparator: L'instance actuelle après transformation.
        """
        if "steps" in self.data.columns:
            logger.info("Traitement de la colonne 'steps'.")
            
            def process_steps_stemming(list_steps: Any) -> str:
                try:
                    list_text = ast.literal_eval(list_steps)
                    text = " ".join(list_text)
                    tokens = nltk.word_tokenize(text)
                    filtered_tokens = [
                        word for word in tokens if word.isalnum() and (word.isdigit() or word not in self.stop_words)
                    ]
                    processed_tokens = [
                        self.stemmer.stem(word) if word.isalpha() else word for word in filtered_tokens
                    ]
                    return " ".join(processed_tokens)
                except Exception as e:
                    logger.error(f"Erreur lors du traitement des étapes : {e}")
                    return ""

            self.data["steps"] = self.data["steps"].apply(process_steps_stemming)
        else:
            logger.warning("La colonne 'steps' n'est pas présente dans le DataFrame.")
        return self

    def process_name(self) -> "VectorizerPreparator":
        """
        Traite la colonne 'name' en enlevant les stop words et la ponctuation, 
        et en appliquant le stemming.

        Returns:
            VectorizerPreparator: L'instance actuelle après transformation.
        """
        if "name" in self.data.columns:
            logger.info("Traitement de la colonne 'name'.")

            def process_name_stemming(string_name: str) -> str:
                try:
                    tokens = nltk.word_tokenize(string_name)
                    filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stop_words]
                    stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
                    return " ".join(stemmed_tokens)
                except Exception as e:
                    logger.error(f"Erreur lors du traitement du nom : {e}")
                    return ""

            self.data["name"] = self.data["name"].apply(process_name_stemming)
        else:
            logger.warning("La colonne 'name' n'est pas présente dans le DataFrame.")
        return self

    def process_tags(self) -> "VectorizerPreparator":
        """
        Transforme la colonne 'tags' en une chaîne de caractères.

        Returns:
            VectorizerPreparator: L'instance actuelle après transformation.
        """
        if "tags" in self.data.columns:
            logger.info("Traitement de la colonne 'tags'.")
            self.data["tags"] = self.data["tags"].apply(lambda x: " ".join(ast.literal_eval(x)))
        else:
            logger.warning("La colonne 'tags' n'est pas présente dans le DataFrame.")
        return self

    def get_prepared_data(self) -> pd.DataFrame:
        """
        Retourne le DataFrame préparé.

        Returns:
            pd.DataFrame: Le DataFrame transformé et prêt pour la vectorisation.
        """
        logger.info("Récupération des données préparées.")
        return self.data

