import pandas as pd
import streamlit as st
from recipe_app import RecipeApp
from app_manager import AppManager

# Charger les datasets avec l'option low_memory=False
merged_clean_df = pd.read_csv('base_light_V3.csv', low_memory=False)
ingredients_part1 = pd.read_csv('id_ingredients_up_to_207226.csv', low_memory=False)
ingredients_part2 = pd.read_csv('id_ingredients_up_to_537716.csv', low_memory=False)

manager = AppManager()


# Ajouter un fond d'écran (image à partir de l'URL)
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://urlr.me/MzRucC");
    background-size: cover;
    background-repeat: no-repeat; 
    background-attachment: fixed;
}
body {
    color: #8B4513; /* Change tout le texte au centre en marron */
}
h1, h2, h3, h4, h5, h6 {
    color: #8B4513; /* Couleur du texte */
    background-color: rgba(255, 255, 255, 0.8); /* Rectangle blanc semi-transparent */
    padding: 10px; /* Espacement interne */
    border-radius: 10px; /* Coins arrondis */
    display: inline-block; /* Ajuster la taille du rectangle au texte */
    text-align: center; /* Centrer le texte */
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); /* Ajouter une ombre légère */
}
.sidebar .block-container h1, .sidebar .block-container h2, .sidebar .block-container h3, .sidebar .block-container h4 {
    font-size: larger; /* Augmente la taille des textes dans la barre latérale */
}
.stTextContainer {
    color: #8B4513; /* Couleur du texte dans les rectangles */
    background-color: rgba(255, 255, 255, 0.8); /* Rectangle blanc semi-transparent */
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 15px;
    box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); /* Ombre légère */
}
.sidebar-content {
    visibility: visible; /* Ouvre le menu de gauche par défaut */
    opacity: 1;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Style pour personnaliser le menu et agrandir la flèche
menu_style = '''
<style>
.sidebar .block-container label {
    font-weight: bold;
    font-style: italic;
    font-size: large; /* Augmente la taille du texte Menu */
}
.sidebar .block-container .radio {
    font-size: larger; /* Augmente la taille des options Accueil et Idée recette */
}
.css-1v0mbdj .stSelectbox div[role="combobox"] {
    font-size: larger; /* Augmente la taille du texte dans le menu déroulant */
}
.css-1v0mbdj .stSelectbox div[role="combobox"]::after {
    content: " ⬇️ Chercher ici !"; /* Ajoute le texte "Chercher ici" avec une grande flèche */
    font-size: larger;
    font-weight: bold;
    color: #8B4513; /* Marron pour correspondre au thème */
}
</style>
'''
st.markdown(menu_style, unsafe_allow_html=True)

# Créer le menu pour changer de page
menu = st.sidebar.radio("**_Menu_**", ["Accueil", "Idée recette !","Représentation des recettes"], index=0)

if menu == "Accueil":
    # Titre de l'application
    st.title("Recherche des meilleures recettes")

    # Ajout de filtres interactifs (placés à droite)
    with st.sidebar:
        st.header("Filtres")
        selected_palmares = st.multiselect(
            "Filtrer par palmarès",
            options=merged_clean_df['palmarès'].unique(),
            default=merged_clean_df['palmarès'].unique()
        )

        selected_steps_category = st.multiselect(
            "Filtrer par catégorie de steps",
            options=merged_clean_df['steps_category'].unique(),
            default=merged_clean_df['steps_category'].unique()
        )

    # Appliquer les filtres
    filtered_df = merged_clean_df[
        (merged_clean_df['palmarès'].isin(selected_palmares)) & 
        (merged_clean_df['steps_category'].isin(selected_steps_category))
    ]

    # Menu déroulant pour sélectionner un contributor_id
    unique_contributor_ids = sorted(filtered_df['contributor_id'].unique())
    contributor_id = st.selectbox("Sélectionnez un contributor_id :", options=unique_contributor_ids)

    # Vérifier si un contributor_id est sélectionné
    if contributor_id:
        # Filtrer les données pour le contributor_id
        contributor_recipes = filtered_df[filtered_df['contributor_id'] == contributor_id]

        if not contributor_recipes.empty:
            # Limiter à 20 recettes maximum
            top_20_ids = contributor_recipes['id'].head(20)

            # Filtrer les fichiers d'ingrédients pour les IDs sélectionnés
            relevant_ingredients_part1 = ingredients_part1[ingredients_part1['id'].isin(top_20_ids)]
            relevant_ingredients_part2 = ingredients_part2[ingredients_part2['id'].isin(top_20_ids)]

            # Fusionner les deux datasets d'ingrédients
            ingredients_combined = pd.concat([relevant_ingredients_part1, relevant_ingredients_part2])

            # Fusionner avec les données principales
            merged_data = pd.merge(contributor_recipes, ingredients_combined, on='id', how='inner')

            # Sélectionner les colonnes importantes
            display_data = merged_data[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category', 'ingredients']].head(20)

            # Afficher les données
            st.subheader(f"Recettes pour le contributor_id {contributor_id} (max 20 recettes)")
            st.dataframe(display_data)
        else:
            st.warning("Aucune recette trouvée pour ce contributor_id.")

    # Afficher les données filtrées avec les filtres interactifs
    st.subheader("Recettes filtrées selon vos critères")
    st.dataframe(filtered_df[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']].head(10))

elif menu == "Idée recette !":
    # Titre de la page
    st.title("Idée recette !")
    st.markdown('<div class="stTextContainer">Ici, vous pouvez explorer de nouvelles idées de recettes.</div>', unsafe_allow_html=True)

    # Instancier et exécuter RecipeApp
    app = RecipeApp()
    app.run()

elif menu == 'Représentation des recettes':
    # Instancier le gestionnaire d'application
    app = RecipeApp()

      # Étape 1 : Fusionner les fichiers d'ingrédients
    # Concaténer les fichiers, en supposant qu'ils ont les mêmes colonnes
    ingredients = pd.concat([ingredients_part1, ingredients_part2], ignore_index=True)

    # Assurez-vous que les colonnes 'ingredients' sont des listes (non du texte brut)
    # Supposons que les ingrédients sont encodés en texte brut dans les fichiers CSV :
    ingredients['ingredients'] = ingredients['ingredients'].apply(eval)

    # Étape 2 : Fusionner avec le DataFrame principal sur la clé 'id'
    merged_clean_df = pd.merge(
        merged_clean_df,
        ingredients,
        on='id',
        how='inner'
    )

    # Étape 3 : Nettoyer les colonnes inutiles
    merged_clean_df = merged_clean_df[['id', 'name', 'ingredients', 'contributor_id']]
    
    # Appliquer les filtres
    unique_contributor_ids = sorted(merged_clean_df['contributor_id'].unique())
    
    st.subheader("Visualisation de mes recettes")
    
    # Ajouter un espacement de 4 lignes
    st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    
    contributor_id = st.selectbox("Sélectionnez un contributor_id :", options=unique_contributor_ids)

    if contributor_id:
        # Filtrer les données pour le contributor_id
        filtered_recipes = merged_clean_df[merged_clean_df['contributor_id'] == contributor_id]


        selected_macros = st.multiselect(
            "Sélectionnez les ingrédients macro parmi la liste triée :",
            options=app.ingredients_macro
        )

        # Assurez-vous que les ingrédients sont sélectionnés avant d'appeler t-SNE
        if selected_macros:
            manager.perform_tsne_with_streamlit(
                recipes=filtered_recipes,  # Passez tout le DataFrame ici
                selected_ingredients=selected_macros,  # Ingrédients sélectionnés
                contributor_id=contributor_id  # ID du contributeur sélectionné
            )
        else:
            st.warning("Veuillez sélectionner au moins un ingrédient.")

    

