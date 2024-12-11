import logging
import os
from typing import List
import pandas as pd

# Assurez-vous que le dossier logs existe
os.makedirs("logs", exist_ok=True)

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


class DatasetSplitter:
    """
    Classe pour diviser un DataFrame en plusieurs fichiers ou parties, selon différentes règles.
    """

    def __init__(self, data: pd.DataFrame, output_dir: str):
        """
        Initialise une instance de DatasetSplitter.

        Args:
            data (pd.DataFrame): DataFrame à diviser.
            output_dir (str): Dossier où sauvegarder les fichiers résultants.
        """
        self.data = data
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Dossier créé : {self.output_dir}")

    def split_by_column(self, columns_to_split: List[str]) -> None:
        """
        Divise le DataFrame en plusieurs fichiers, en conservant l'ID et les colonnes spécifiées.

        Args:
            columns_to_split (List[str]): Liste des colonnes à utiliser pour la division.

        Raises:
            ValueError: Si une colonne de `columns_to_split` n'est pas dans le DataFrame.
        """
        for column in columns_to_split:
            if column not in self.data.columns:
                logger.error(f"Colonne manquante : {column}")
                raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

            subset = self.data[["id", column]].copy()
            output_path = os.path.join(self.output_dir, f"pp_recipes_{column}.csv")
            subset.to_csv(output_path, index=False)
            logger.info(f"Fichier sauvegardé pour la colonne '{column}' : {output_path}")

    def split_by_numeric_columns(self, numeric_columns: List[str]) -> None:
        """
        Divise le DataFrame en ne gardant que les colonnes numériques spécifiées et l'ID.

        Args:
            numeric_columns (List[str]): Liste des colonnes numériques à conserver.

        Raises:
            ValueError: Si une colonne de `numeric_columns` n'est pas dans le DataFrame.
        """
        for column in numeric_columns:
            if column not in self.data.columns:
                logger.error(f"Colonne manquante : {column}")
                raise ValueError(f"La colonne '{column}' n'existe pas dans le DataFrame.")

        numeric_subset = self.data[["id"] + numeric_columns].copy()
        output_path = os.path.join(self.output_dir, "pp_recipes_numerics.csv")
        numeric_subset.to_csv(output_path, index=False)
        logger.info(f"Fichier sauvegardé pour colonnes numériques : {output_path}")

    def split_into_parts(self, input_file: str, num_parts: int) -> None:
        """
        Divise un fichier CSV en plusieurs parties égales en fonction des lignes.

        Args:
            input_file (str): Chemin du fichier CSV d'entrée.
            num_parts (int): Nombre de parties à créer.

        Raises:
            FileNotFoundError: Si le fichier d'entrée n'existe pas.
            ValueError: Si `num_parts` est inférieur à 1.
        """
        if not os.path.exists(input_file):
            logger.error(f"Fichier introuvable : {input_file}")
            raise FileNotFoundError(f"Le fichier '{input_file}' n'existe pas.")

        if num_parts < 1:
            logger.error(f"Nombre de parties invalide : {num_parts}")
            raise ValueError("Le nombre de parties doit être au moins égal à 1.")

        data = pd.read_csv(input_file)
        part_size = len(data) // num_parts

        for i in range(num_parts):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < num_parts - 1 else len(data)
            subset = data.iloc[start_idx:end_idx].copy()

            base_name = os.path.basename(input_file).replace(".csv", "")
            output_path = os.path.join(self.output_dir, f"{base_name}_part_{i + 1}.csv")
            subset.to_csv(output_path, index=False)
            logger.info(f"Fichier sauvegardé : {output_path}")


