import logging
from typing import Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import csr_matrix
import os

# Assurez-vous que le dossier logs existe
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,  # Pour capturer INFO, ERROR, et DEBUG
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/debug.log", mode="a"),  # Logs généraux
        logging.FileHandler("logs/error.log", mode="a"),  # Logs des erreurs
        logging.StreamHandler()  # Afficher dans la console
    ]
)

logger = logging.getLogger(__name__)

class Vectorizer:
    """
    Une classe utilitaire pour gérer la vectorisation des colonnes textuelles
    à l'aide de TF-IDF ou Bag of Words (BoW).
    """

    @staticmethod
    def tfidf_vectorize(column_data: pd.Series) -> Tuple[csr_matrix, TfidfVectorizer]:
        """
        Effectue une vectorisation TF-IDF sur les données textuelles d'une colonne.

        Args:
            column_data (pd.Series): Une colonne Pandas contenant les données textuelles.

        Returns:
            Tuple[csr_matrix, TfidfVectorizer]: La matrice TF-IDF sous forme creuse (sparse)
            et l'instance du TfidfVectorizer utilisée pour la transformation.

        Raises:
            ValueError: Si column_data est vide ou non valide.
        """
        if column_data.empty:
            logger.error("La colonne fournie pour la vectorisation TF-IDF est vide.")
            raise ValueError("La colonne fournie est vide.")

        logger.info("Début de la vectorisation TF-IDF.")
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(column_data)
        logger.info("Vectorisation TF-IDF terminée avec succès.")
        return tfidf_matrix, vectorizer

    @staticmethod
    def bow_vectorize(column_data: pd.Series) -> Tuple[csr_matrix, CountVectorizer]:
        """
        Effectue une vectorisation Bag of Words (BoW) sur les données textuelles d'une colonne.

        Args:
            column_data (pd.Series): Une colonne Pandas contenant les données textuelles.

        Returns:
            Tuple[csr_matrix, CountVectorizer]: La matrice BoW sous forme creuse (sparse)
            et l'instance du CountVectorizer utilisée pour la transformation.

        Raises:
            ValueError: Si column_data est vide ou non valide.
        """
        if column_data.empty:
            logger.error("La colonne fournie pour la vectorisation BoW est vide.")
            raise ValueError("La colonne fournie est vide.")

        logger.info("Début de la vectorisation BoW.")
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(column_data)
        logger.info("Vectorisation BoW terminée avec succès.")
        return bow_matrix, vectorizer

