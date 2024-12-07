import pandas as pd
import streamlit as st
import ast

class RecipeApp:
    def __init__(self):
        self.ingredients_macro = sorted([
            "butter", "sugar", "onion", "water", "eggs", "oil", "flour",
            "milk", "garlic", "pepper", "baking powder", "egg", "cheese",
            "lemon juice", "baking soda", "vanilla", "cinnamon", "tomatoe",
            "sour cream", "honey", "cream cheese", "celery", "soy sauce",
            "mayonnaise", "paprika", "chicken", "worcestershire sauce",
            "parsley", "cornstarch", "carrot", "chili", "bacon", "potatoe"
        ])
        # Charger les données
        self.recipes_clean = self.load_main_data()

    @staticmethod
    @st.cache_data
    def load_main_data():
        """Charge les données principales depuis base_light_V3."""
        return pd.read_csv('base_light_V3.csv', usecols=['id', 'name', 'ingredients', 'contributor_id'], low_memory=False)

    def filter_recipes(self, selected_ingredients):
        """
        Filtre les recettes contenant tous les ingrédients sélectionnés
        (recherche partielle sur les noms d'ingrédients).
        """
        if not selected_ingredients:
            return pd.DataFrame()  # Si aucun ingrédient sélectionné, renvoyer un DataFrame vide

        def contains_all_selected_ingredients(recipe_ingredients):
            """Vérifie si les ingrédients de la recette contiennent tous les ingrédients sélectionnés."""
            recipe_ingredients = set(ast.literal_eval(recipe_ingredients))
            return all(
                any(selected in ingredient for ingredient in recipe_ingredients)
                for selected in selected_ingredients
            )

        # Appliquer le filtre
        filtered_recipes = self.recipes_clean[
            self.recipes_clean['ingredients'].apply(contains_all_selected_ingredients)
        ]
        return filtered_recipes.head(10)  # Limiter aux 10 premières recettes

    def display_macro_ingredients_menu(self):
        """Affiche un menu déroulant pour choisir plusieurs ingrédients macro."""
        st.subheader("Choisissez vos ingrédients macro")
        selected_macros = st.multiselect(
            "Sélectionnez les ingrédients macro parmi la liste triée :",
            options=self.ingredients_macro
        )
        st.write(f"Vous avez sélectionné : {', '.join(selected_macros) if selected_macros else 'Aucun ingrédient sélectionné.'}")
        return selected_macros

    def display_filtered_recipes(self, selected_ingredients):
        """Affiche les recettes filtrées en fonction des ingrédients sélectionnés."""
        st.subheader("Résultats des recettes filtrées")
        filtered_recipes = self.filter_recipes(selected_ingredients)
        if not filtered_recipes.empty:
            st.write(f"Voici les 10 premières recettes contenant tous les ingrédients sélectionnés :")
            st.dataframe(filtered_recipes[['id', 'name', 'contributor_id']])
        else:
            st.warning("Aucune recette ne correspond à vos ingrédients sélectionnés.")

    def run(self):
        """Exécute l'application Streamlit."""
        st.title("Recipe App")
        st.write("Explorez les recettes en fonction des ingrédients sélectionnés.")

        # Étape 1 : Sélection des ingrédients macro
        selected_ingredients = self.display_macro_ingredients_menu()

        # Étape 2 : Afficher les recettes filtrées
        self.display_filtered_recipes(selected_ingredients)

# Exécution de l'application
if __name__ == "__main__":
    app = RecipeApp()
    app.run()
