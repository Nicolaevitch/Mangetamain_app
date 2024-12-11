import pandas as pd
import streamlit as st
import logging
from typing import List, Optional
from src.recipe_app.recipe_app import RecipeApp
from src.app_manager.app_manager import AppManager
from src.FindingCloseRecipes.run_recipe_finder import run_recipe_finder  # Import de la fonction pour la recherche de recettes proches

# Configurer les loggers
logging.basicConfig(level=logging.DEBUG, filename='debug.log', filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')
error_logger = logging.getLogger('error_logger')
error_handler = logging.FileHandler('error.log')
error_handler.setLevel(logging.ERROR)
error_logger.addHandler(error_handler)

class RecipeDashboard:
    def __init__(self):
        """
        Initialisation de la classe RecipeDashboard.
        """
        self.merged_clean_df: Optional[pd.DataFrame] = None
        self.ingredients_part1: Optional[pd.DataFrame] = None
        self.ingredients_part2: Optional[pd.DataFrame] = None
        self.manager = AppManager()
        self.load_data()

    def load_data(self):
        """
        Charge les datasets nécessaires pour l'application.
        """
        try:
            self.merged_clean_df = pd.read_csv('data/base_light_V3.csv', low_memory=False)
            self.ingredients_part1 = pd.read_csv('data/id_ingredients_up_to_207226.csv', low_memory=False)
            self.ingredients_part2 = pd.read_csv('data/id_ingredients_up_to_537716.csv', low_memory=False)
            logging.info("Les données ont été chargées avec succès.")
        except Exception as e:
            error_logger.error(f"Erreur lors du chargement des données : {e}")
            st.error(f"Erreur lors du chargement des données : {e}")
            st.stop()

    def add_custom_styles(self):
        """
        Ajoute des styles personnalisés à l'application Streamlit.
        """
        page_bg_img = '''
        <style>
        .stApp {
            background-image: url("https://urlr.me/MzRucC");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        body {
            color: #8B4513;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #8B4513;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 10px;
            display: inline-block;
            text-align: center;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
        }
        .sidebar .block-container label {
            font-weight: bold;
            font-style: italic;
            font-size: large;
        }
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)

    def display_home_page(self):
        """
        Affiche la page d'accueil sans les filtres sur la barre latérale.
        """
        st.title("Bienvenue sur ton profil de recettes !")

        try:
            filtered_df = self.merged_clean_df

            unique_contributor_ids = sorted(filtered_df['contributor_id'].unique())
            contributor_id = st.selectbox("Sélectionnez un contributor_id :", options=unique_contributor_ids)

            if contributor_id:
                self.display_contributor_data(filtered_df, contributor_id)
        except Exception as e:
            error_logger.error(f"Erreur lors de l'affichage de la page d'accueil : {e}")
            st.error("Une erreur s'est produite lors de l'affichage de la page d'accueil.")

    def display_contributor_data(self, filtered_df: pd.DataFrame, contributor_id: int):
        """
        Affiche les données d'un contributor_id sélectionné.

        Args:
            filtered_df (pd.DataFrame): DataFrame filtré avec les données des contributeurs.
            contributor_id (int): Identifiant du contributeur à afficher.
        """
        try:
            contributor_recipes = filtered_df[filtered_df['contributor_id'] == contributor_id]
            if contributor_recipes.empty:
                st.warning("Aucune recette trouvée pour ce contributor_id.")
                return

            palmares = contributor_recipes['palmarès'].iloc[0]
            recipe_count = contributor_recipes['id'].nunique()
            average_rating = contributor_recipes['average_rating'].mean()

            st.markdown(f"""
            <style>
            .kpi-container {{
                display: flex;
                gap: 20px;
                justify-content: center;
                margin-bottom: 20px;
            }}
            .kpi-box {{
                background-color: #f4f4f4;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                width: 200px;
            }}
            .kpi-title {{
                font-size: 18px;
                font-weight: bold;
                color: #8B4513;
            }}
            .kpi-value {{
                font-size: 24px;
                font-weight: bold;
                margin-top: 5px;
                color: #8B4513;
            }}
            </style>

            <div class="kpi-container">
                <div class="kpi-box">
                    <div class="kpi-title">Palmarès</div>
                    <div class="kpi-value">{palmares}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-title">Total Recettes</div>
                    <div class="kpi-value">{recipe_count}</div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-title">Note Moyenne</div>
                    <div class="kpi-value">{average_rating:.2f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            top_20_ids = contributor_recipes['id'].head(20)
            relevant_ingredients_part1 = self.ingredients_part1[self.ingredients_part1['id'].isin(top_20_ids)]
            relevant_ingredients_part2 = self.ingredients_part2[self.ingredients_part2['id'].isin(top_20_ids)]
            ingredients_combined = pd.concat([relevant_ingredients_part1, relevant_ingredients_part2])

            merged_data = pd.merge(contributor_recipes, ingredients_combined, on='id', how='inner')

            display_data = merged_data[['id', 'name', 'average_rating', 'minutes', 'palmarès', 'steps_category', 'ingredients']].head(20)
            st.subheader(f"Recettes pour le contributor_id {contributor_id} (max 20 recettes)")
            st.dataframe(display_data)

            all_ingredients = merged_data['ingredients'].apply(eval).explode()
            ingredient_counts = all_ingredients.value_counts().head(10).reset_index()
            ingredient_counts.columns = ['Ingredient', 'Count']

            st.subheader(f"Top 10 des ingrédients les plus utilisés par {contributor_id}")
            st.dataframe(ingredient_counts)

        except Exception as e:
            error_logger.error(f"Erreur lors de l'affichage des données du contributeur : {e}")
            st.error("Une erreur s'est produite lors de l'affichage des données du contributeur.")

    def run(self):
        """
        Lance l'application Streamlit.
        """
        try:
            self.add_custom_styles()
            menu = st.sidebar.radio("**_Menu_**", ["Accueil", "Idée recette !", "Représentation des recettes", "Recherche de Recettes Proches"], index=0)

            if menu == "Accueil":
                self.display_home_page()
            elif menu == "Idée recette !":
                app = RecipeApp()
                app.run()
            elif menu == "Représentation des recettes":
                self.display_visualization_page()
            elif menu == "Recherche de Recettes Proches":
                self.display_recipe_search_page()
        except Exception as e:
            error_logger.error(f"Erreur générale de l'application : {e}")
            st.error("Une erreur critique s'est produite dans l'application.")

if __name__ == "__main__":
    dashboard = RecipeDashboard()
    dashboard.run()
