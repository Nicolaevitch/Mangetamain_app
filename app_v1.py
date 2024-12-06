import pandas as pd
import streamlit as st

# Charger les datasets avec l'option low_memory=False
merged_clean_df = pd.read_csv('base_light_V3.csv', low_memory=False)
ingredients_part1 = pd.read_csv('id_ingredients_up_to_207226.csv', low_memory=False)
ingredients_part2 = pd.read_csv('id_ingredients_up_to_537716.csv', low_memory=False)

# Ajouter un fond d'écran (image à partir de l'URL)
page_bg_img = '''
<style>
.stApp {
    background-image: url("https://burst.shopifycdn.com/photos/flatlay-iron-skillet-with-meat-and-other-food.jpg?width=925&format=pjpg&exif=0&iptc=0");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
body {
    color: #D2B48C; /* Change tout le texte au centre en beige légèrement foncé */
}
h1, h2, h3, h4, h5, h6 {
    color: #D2B48C; /* Change aussi la couleur des titres */
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Titre de l'application
st.title("Recherche des meilleures recettes")

# Afficher un top 10 des recettes avec le meilleur score average_rating
st.subheader("Top 10 des recettes avec le meilleur score")
top_10_recipes = merged_clean_df.sort_values(by='average_rating', ascending=False).head(10)
st.dataframe(top_10_recipes[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']])

# Ajout de filtres interactifs
st.sidebar.header("Filtres")
selected_palmares = st.sidebar.multiselect(
    "Filtrer par palmarès",
    options=merged_clean_df['palmarès'].unique(),
    default=merged_clean_df['palmarès'].unique()
)

selected_steps_category = st.sidebar.multiselect(
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
