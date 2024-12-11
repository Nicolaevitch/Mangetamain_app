import pandas as pd
import streamlit as st
import ast
import logging
from typing import List, Optional

# Configurer les loggers
logging.basicConfig(level=logging.DEBUG, filename='logs/debug.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler('logs/error.log')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

class IngredientDataError(Exception):
    """Exception personnalisée pour les erreurs liées aux données des ingrédients."""
    pass

class RecipeApp:
    def __init__(self):
        """Initialise les données de l'application de recettes."""
        self.ingredients_macro: List[str] = sorted([
            "butter", "sugar", "onion", "water", "eggs", "oil", "flour",
            "milk", "garlic", "pepper", "baking powder", "egg", "cheese",
            "lemon juice", "baking soda", "vanilla", "cinnamon", "tomatoe",
            "sour cream", "honey", "cream cheese", "celery", "soy sauce",
            "mayonnaise", "paprika", "chicken", "worcestershire sauce",
            "parsley", "cornstarch", "carrot", "chili", "bacon", "potatoe"
        ])
        self.file_part1: str = 'data/id_ingredients_up_to_207226.csv'
        self.file_part2: str = 'data/id_ingredients_up_to_537716.csv'
        self.main_file: str = 'data/base_light_V3.csv'
        self.recipes_clean: pd.DataFrame = self.load_main_data()

    @staticmethod
    @st.cache_data
    def load_main_data() -> pd.DataFrame:
        """Charge les données principales depuis base_light_V3."""
        try:
            return pd.read_csv('data/base_light_V3.csv', low_memory=False)
        except Exception as e:
            error_logger.error(f"Erreur lors du chargement des données principales : {e}")
            raise IngredientDataError("Impossible de charger les données principales.")

    def get_ingredients_data(self) -> pd.DataFrame:
        """Charge et combine les données des deux fichiers d'ingrédients."""
        try:
            part1_data = pd.read_csv(self.file_part1, usecols=['id', 'ingredients'], low_memory=False)
            part2_data = pd.read_csv(self.file_part2, usecols=['id', 'ingredients'], low_memory=False)
            return pd.concat([part1_data, part2_data])
        except Exception as e:
            error_logger.error(f"Erreur lors du chargement des données des ingrédients : {e}")
            raise IngredientDataError("Erreur lors du chargement des fichiers d'ingrédients.")

    def filter_recipes(self, selected_ingredients: List[str]) -> pd.DataFrame:
        """
        Filtre les recettes contenant tous les ingrédients sélectionnés
        (recherche partielle sur les noms d'ingrédients).

        Args:
            selected_ingredients (List[str]): Liste des ingrédients sélectionnés.

        Returns:
            pd.DataFrame: Recettes filtrées correspondant aux critères.
        """
        if not selected_ingredients:
            return pd.DataFrame()

        ingredients_data = self.get_ingredients_data()

        def contains_all_selected_ingredients(recipe_ingredients: str) -> bool:
            """Vérifie si les ingrédients de la recette contiennent tous les ingrédients sélectionnés."""
            try:
                recipe_ingredients_set = set(ast.literal_eval(recipe_ingredients))
                return all(
                    any(selected in ingredient for ingredient in recipe_ingredients_set)
                    for selected in selected_ingredients
                )
            except (ValueError, SyntaxError) as e:
                error_logger.error(f"Erreur lors de l'analyse des ingrédients : {e}")
                return False

        filtered_ingredients = ingredients_data[
            ingredients_data['ingredients'].apply(contains_all_selected_ingredients)
        ]

        filtered_ids = filtered_ingredients['id'].head(10)
        filtered_recipes = self.recipes_clean[self.recipes_clean['id'].isin(filtered_ids)]
        filtered_recipes = pd.merge(filtered_recipes, filtered_ingredients, on='id', how='left')
        return filtered_recipes

    def display_macro_ingredients_menu(self) -> List[str]:
        """Affiche un menu déroulant pour choisir plusieurs ingrédients macro."""
        return st.multiselect(
            "Sélectionnez les ingrédients parmi la liste triée :",
            options=self.ingredients_macro
        )

    def display_filtered_recipes(self, selected_ingredients: List[str]):
        """Affiche les recettes filtrées en fonction des ingrédients sélectionnés."""
        try:
            filtered_recipes = self.filter_recipes(selected_ingredients)

            if not filtered_recipes.empty:
                info_options = ['id', 'name', 'contributor_id', 'steps_category', 'palmarès', 'ingredients']
                selected_info = st.multiselect(
                    "Choisissez les colonnes à afficher :",
                    options=info_options,
                    default=['id', 'name', 'ingredients']
                )

                if 'ingredients' in selected_info:
                    filtered_recipes['ingredients'] = filtered_recipes['ingredients'].apply(
                        lambda x: "\n".join(x) if isinstance(x, list) else x
                    )

                st.dataframe(
                    filtered_recipes[selected_info],
                    use_container_width=True
                )

                self.display_recipe_details(filtered_recipes, selected_info)
            else:
                st.title("On est pas des cakes !")
        except Exception as e:
            error_logger.error(f"Erreur lors de l'affichage des recettes filtrées : {e}")
            st.error("Une erreur s'est produite lors de l'affichage des recettes.")

    def display_recipe_details(self, filtered_recipes: pd.DataFrame, selected_info: List[str]):
        """
        Affiche les détails d'une recette sélectionnée par ID.

        Args:
            filtered_recipes (pd.DataFrame): Recettes filtrées.
            selected_info (List[str]): Colonnes sélectionnées pour l'affichage.
        """
        selected_recipe_id = st.selectbox(
            "Choisissez une recette par ID :",
            options=filtered_recipes['id']
        )

        selected_recipe_data = filtered_recipes[filtered_recipes['id'] == selected_recipe_id]
        st.subheader("Détails de la recette sélectionnée :")
        st.dataframe(
            selected_recipe_data[selected_info],
            use_container_width=True
        )

    def run(self):
        """Exécute l'application Streamlit."""
        st.title("Qu'est ce que tu as dans ton frigo ?")
        selected_ingredients = self.display_macro_ingredients_menu()
        self.display_filtered_recipes(selected_ingredients)

if __name__ == "__main__":
    try:
        app = RecipeApp()
        app.run()
    except Exception as e:
        error_logger.critical(f"Erreur critique lors de l'exécution de l'application : {e}")
