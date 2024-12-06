import pandas as pd
import streamlit as st

class RecipeApp:
    def __init__(self):
        # Chemins vers les fichiers
        self.main_file = 'base_light_V3.csv'
        self.file_part1 = 'id_ingredients_up_to_207226.csv'
        self.file_part2 = 'id_ingredients_up_to_537716.csv'

        # Charger les données
        self.recipes_clean = self.load_main_data()

    @staticmethod
    @st.cache_data
    def load_main_data():
        """Charge les données principales depuis base_light_V3."""
        return pd.read_csv('base_light_V3', usecols=['id', 'name', 'average_rating', 'contributor_id'])

    def get_ingredients_by_ids(self, recipe_ids):
        """Récupère les ingrédients pour une liste d'IDs de recette depuis les fichiers spécifiques."""
        part1_ids = [rid for rid in recipe_ids if rid <= 207226]
        part2_ids = [rid for rid in recipe_ids if rid > 207226]

        # Charger les fichiers d'ingrédients
        part1_data = pd.read_csv(self.file_part1, usecols=['id', 'ingredients'], low_memory=False)
        part2_data = pd.read_csv(self.file_part2, usecols=['id', 'ingredients'], low_memory=False)

        # Filtrer les données pour les IDs demandés
        filtered_part1 = part1_data[part1_data['id'].isin(part1_ids)]
        filtered_part2 = part2_data[part2_data['id'].isin(part2_ids)]

        # Combiner les données
        combined_data = pd.concat([filtered_part1, filtered_part2])
        return combined_data

    def display_recipe_by_id(self, recipe_id):
        """Affiche une recette spécifique par ID."""
        recipe = self.recipes_clean[self.recipes_clean['id'] == recipe_id]
        if recipe.empty:
            st.error(f"Aucune recette trouvée avec l'ID {recipe_id}.")
            return

        # Récupérer les ingrédients pour la recette
        ingredients = self.get_ingredients_by_ids([recipe_id])
        recipe_row = recipe.iloc[0]
        ingredients_row = ingredients[ingredients['id'] == recipe_id]

        st.subheader(f"Recette : {recipe_row['name']}")
        st.write(f"**Note Moyenne**: {recipe_row['average_rating']}")
        st.write(f"**Contributeur**: {recipe_row['contributor_id']}")
        st.write(f"**Ingrédients**: {', '.join(ingredients_row['ingredients'].iloc[0]) if not ingredients_row.empty else 'Ingrédients non trouvés.'}")

    def run(self):
        """Exécute l'application Streamlit."""
        st.title("Recipe App")
        st.write("Recherchez une recette en fonction de son ID.")

        recipe_id = st.text_input("Entrez l'ID de la recette :", "")
        if recipe_id.isdigit():
            recipe_id = int(recipe_id)
            self.display_recipe_by_id(recipe_id)
        else:
            st.warning("Veuillez entrer un ID valide.")
