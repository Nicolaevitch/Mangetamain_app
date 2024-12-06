import pandas as pd
import streamlit as st
import ast

class RecipeApp:
    def __init__(self):
        self.ingredients_macro = [
            "butter", "sugar", "onion", "water", "eggs", "oil", "flour",
            "milk", "garlic", "pepper", "baking powder", "egg", "cheese",
            "lemon juice", "baking soda", "vanilla", "cinnamon", "tomatoe",
            "sour cream", "honey", "cream cheese", "celery", "soy sauce",
            "mayonnaise", "paprika", "chicken", "worcestershire sauce",
            "parsley", "cornstarch", "carrot", "chili", "bacon", "potatoe"
        ]
        # Chemins vers les fichiers
        self.file_part1 = 'id_ingredients_up_to_207226.csv'
        self.file_part2 = 'id_ingredients_up_to_537716.csv'
        self.recipes_clean, self.interactions_clean = self.load_data()

    @staticmethod
    @st.cache_data
    def load_data():
        """Charge les données nécessaires depuis les fichiers CSV."""
        recipes = pd.read_csv('clean_recipes.csv', usecols=['id', 'name', 'ingredients', 'contributor_id'])
        interactions = pd.read_csv('clean_interactions.csv', usecols=['recipe_id', 'rating'])
        recipes['ingredients'] = recipes['ingredients'].apply(ast.literal_eval)
        return recipes, interactions

    def get_ingredients_by_ids(self, recipe_ids):
        """Récupère les ingrédients pour une liste d'IDs de recette depuis les fichiers."""
        part1_ids = [rid for rid in recipe_ids if rid <= 207226]
        part2_ids = [rid for rid in recipe_ids if rid > 207226]
        part1_data = pd.read_csv(self.file_part1, low_memory=False)
        part2_data = pd.read_csv(self.file_part2, low_memory=False)
        filtered_part1 = part1_data[part1_data['id'].isin(part1_ids)]
        filtered_part2 = part2_data[part2_data['id'].isin(part2_ids)]
        combined_data = pd.concat([filtered_part1, filtered_part2])
        return combined_data[['id', 'ingredients']]

    def run(self):
        st.title("Recipe App")
        st.write("Ceci est une interface pour la gestion des recettes.")
