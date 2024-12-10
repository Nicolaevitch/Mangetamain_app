import pandas as pd
import streamlit as st
from src.recipe_app import RecipeApp
from src.app_manager import AppManager


class RecipeDashboard:
    def __init__(self):
        self.merged_clean_df = None
        self.ingredients_part1 = None
        self.ingredients_part2 = None
        self.manager = AppManager()
        self.load_data()

    def load_data(self):
        """Charge les datasets nécessaires."""
        try:
            self.merged_clean_df = pd.read_csv('base_light_V3.csv', low_memory=False)
            self.ingredients_part1 = pd.read_csv('id_ingredients_up_to_207226.csv', low_memory=False)
            self.ingredients_part2 = pd.read_csv('id_ingredients_up_to_537716.csv', low_memory=False)
        except Exception as e:
            st.error(f"Erreur lors du chargement des données : {e}")
            st.stop()

    def add_custom_styles(self):
        """Ajoute des styles personnalisés à l'application."""
        # Ajouter un fond d'écran
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
        """Affiche la page d'accueil avec les filtres et les données filtrées."""
        st.title("Bienvenue sur ton profil de recettes !")

        # Ajouter des filtres dans la barre latérale
        with st.sidebar:
            st.header("Filtres")
            selected_palmares = st.multiselect(
                "Filtrer par palmarès",
                options=self.merged_clean_df['palmarès'].unique(),
                default=self.merged_clean_df['palmarès'].unique()
            )
            selected_steps_category = st.multiselect(
                "Filtrer par catégorie de steps",
                options=self.merged_clean_df['steps_category'].unique(),
                default=self.merged_clean_df['steps_category'].unique()
            )

        # Appliquer les filtres
        filtered_df = self.merged_clean_df[
            (self.merged_clean_df['palmarès'].isin(selected_palmares)) &
            (self.merged_clean_df['steps_category'].isin(selected_steps_category))
        ]

        # Menu déroulant pour sélectionner un contributor_id
        unique_contributor_ids = sorted(filtered_df['contributor_id'].unique())
        contributor_id = st.selectbox("Sélectionnez un contributor_id :", options=unique_contributor_ids)

        if contributor_id:
            self.display_contributor_data(filtered_df, contributor_id)

    def display_contributor_data(self, filtered_df, contributor_id):
        """Affiche les données d'un contributor_id sélectionné."""
        contributor_recipes = filtered_df[filtered_df['contributor_id'] == contributor_id]
        if contributor_recipes.empty:
            st.warning("Aucune recette trouvée pour ce contributor_id.")
            return

        # Calcul des KPI
        palmares = contributor_recipes['palmarès'].iloc[0]
        recipe_count = contributor_recipes['id'].nunique()
        average_rating = contributor_recipes['average_rating'].mean()

        # Affichage des bulles avec les trois informations
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

        # Limiter à 20 recettes maximum
        top_20_ids = contributor_recipes['id'].head(20)
        relevant_ingredients_part1 = self.ingredients_part1[self.ingredients_part1['id'].isin(top_20_ids)]
        relevant_ingredients_part2 = self.ingredients_part2[self.ingredients_part2['id'].isin(top_20_ids)]
        ingredients_combined = pd.concat([relevant_ingredients_part1, relevant_ingredients_part2])

        # Fusionner avec les données principales
        merged_data = pd.merge(contributor_recipes, ingredients_combined, on='id', how='inner')

        # Afficher les données principales
        display_data = merged_data[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category', 'ingredients']].head(20)
        st.subheader(f"Recettes pour le contributor_id {contributor_id} (max 20 recettes)")
        st.dataframe(display_data)

        # Calcul des ingrédients les plus utilisés
        all_ingredients = merged_data['ingredients'].apply(eval).explode()
        ingredient_counts = all_ingredients.value_counts().head(10).reset_index()
        ingredient_counts.columns = ['Ingredient', 'Count']

        # Afficher le tableau des ingrédients
        st.subheader(f"Top 10 des ingrédients les plus utilisés par {contributor_id}")
        st.dataframe(ingredient_counts)


    def display_visualization_page(self):
        """Affiche la page de visualisation des recettes."""
        st.subheader("Visualisation de mes recettes")

        # Étape 1 : Fusionner les fichiers d'ingrédients
        ingredients = pd.concat([self.ingredients_part1, self.ingredients_part2], ignore_index=True)

        # Convertir les colonnes en listes (si nécessaire)
        ingredients['ingredients'] = ingredients['ingredients'].apply(eval)

        # Étape 2 : Fusionner avec les données principales
        merged_clean_df = pd.merge(self.merged_clean_df, ingredients, on='id', how='inner')

        # Étape 3 : Nettoyer les colonnes inutiles
        merged_clean_df = merged_clean_df[['id', 'name', 'ingredients', 'contributor_id']]

        # Sélectionner un contributor_id
        unique_contributor_ids = sorted(merged_clean_df['contributor_id'].unique())
        contributor_id = st.selectbox("Sélectionnez un contributor_id :", options=unique_contributor_ids)

        if contributor_id:
            # Filtrer les recettes pour le contributor_id
            filtered_recipes = merged_clean_df[merged_clean_df['contributor_id'] == contributor_id]

            # Sélectionner les macros
            app = RecipeApp()
            selected_macros = st.multiselect(
                "Sélectionnez les ingrédients macro parmi la liste triée :",
                options=app.ingredients_macro
            )

            if selected_macros:
                self.manager.perform_tsne_with_streamlit(
                    recipes=filtered_recipes,
                    selected_ingredients=selected_macros,
                    contributor_id=contributor_id
                )
            else:
                st.warning("Veuillez sélectionner au moins un ingrédient.")

    def run(self):
        """Lance l'application Streamlit."""
        self.add_custom_styles()
        menu = st.sidebar.radio("**_Menu_**", ["Accueil", "Idée recette !", "Représentation des recettes"], index=0)

        if menu == "Accueil":
            self.display_home_page()
        elif menu == "Idée recette !":
            app = RecipeApp()
            app.run()
        elif menu == "Représentation des recettes":
            self.display_visualization_page()


if __name__ == "__main__":
    dashboard = RecipeDashboard()
    dashboard.run()
