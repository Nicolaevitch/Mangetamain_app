import pandas as pd
import os


class DatasetSplitter:
    def __init__(self, data: pd.DataFrame, output_dir: str):
        """
        Classe pour diviser un DataFrame en plusieurs fichiers en fonction des colonnes spécifiées.

        :param data: DataFrame à diviser.
        :param output_dir: Dossier où sauvegarder les fichiers résultants.
        """
        self.data = data
        self.output_dir = output_dir

    def split_by_column(self, columns_to_split: list):
        """
        Divise le DataFrame en plusieurs fichiers, en conservant l'ID et la colonne spécifiée.
        :param columns_to_split: Liste des colonnes à utiliser pour la division.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for column in columns_to_split:
            if column in self.data.columns:
                # Créer un DataFrame avec la colonne et l'ID
                subset = self.data[["id", column]].copy()
                # Sauvegarder dans un fichier
                output_path = os.path.join(self.output_dir, f"pp_recipes_{column}.csv")
                subset.to_csv(output_path, index=False)
                print(f"Fichier sauvegardé : {output_path}")
            else:
                print(f"Colonne non trouvée : {column}")

    def split_by_numeric_columns(self, numeric_columns: list):
        """
        Divise le DataFrame en ne gardant que les colonnes numériques spécifiées et l'ID.
        :param numeric_columns: Liste des colonnes numériques à conserver.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Filtrer les colonnes numériques
        numeric_subset = self.data[["id"] + numeric_columns].copy()
        output_path = os.path.join(self.output_dir, "pp_recipes_numerics.csv")
        numeric_subset.to_csv(output_path, index=False)
        print(f"Fichier sauvegardé pour colonnes numériques : {output_path}")

    def split_into_parts(self, input_file: str, num_parts: int):
        """
        Divise un fichier CSV en plusieurs parties égales en fonction des lignes.
        :param input_file: Chemin du fichier CSV d'entrée.
        :param num_parts: Nombre de parties à créer.
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Charger le fichier CSV
        data = pd.read_csv(input_file)

        # Calculer le nombre de lignes par partie
        part_size = len(data) // num_parts
        for i in range(num_parts):
            # Déterminer les indices de début et de fin pour la partie actuelle
            start_idx = i * part_size
            end_idx = (i + 1) * part_size if i < num_parts - 1 else len(data)

            # Extraire la sous-partie
            subset = data.iloc[start_idx:end_idx].copy()
            # Générer un nouveau nom de fichier
            base_name = os.path.basename(input_file).replace(".csv", "")
            output_path = os.path.join(self.output_dir, f"{base_name}_part_{i + 1}.csv")
            subset.to_csv(output_path, index=False)
            print(f"Fichier sauvegardé : {output_path}")

