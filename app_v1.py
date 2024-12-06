import pandas as pd
import streamlit as st

# Charger le dataset
merged_clean_df = pd.read_csv('base_light_V3.csv')

# Ajouter un fond d'écran (image1)
page_bg_img = '''
<style>
.stApp {
    background-image: url("image1");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
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

# Champ d'entrée pour le contributor_id
contributor_id = st.text_input("Entrez un contributor_id :", "")

# Vérifier si un contributor_id est saisi
if contributor_id:
    try:
        # Convertir le contributor_id en entier
        contributor_id = int(contributor_id)
        
        # Filtrer les données pour le contributor_id
        filtered_data = filtered_df[filtered_df['contributor_id'] == contributor_id]
        
        if not filtered_data.empty:
            # Calculer le total des recettes pour ce contributor_id
            total_recipes = filtered_data.shape[0]

            # Trier par les meilleures notes et sélectionner les colonnes importantes
            top_recipes = filtered_data.sort_values(by='average_rating', ascending=False).head(10)
            top_recipes = top_recipes[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']]
            
            # Afficher les statistiques du contributor
            st.subheader(f"Statistiques pour le contributor avec l'ID {contributor_id}")
            st.write(f"**Total de recettes publiées :** {total_recipes}")

            # Afficher les meilleures recettes du contributor
            st.subheader(f"Top 10 des meilleures recettes pour le contributor_id {contributor_id}")
            st.dataframe(top_recipes)
        else:
            st.warning("Aucune recette trouvée pour ce contributor_id.")
    except ValueError:
        st.error("Veuillez entrer un contributor_id valide.")
else:
    st.info("Veuillez entrer un contributor_id pour afficher les résultats.")

# Afficher les données filtrées avec les filtres interactifs
if not contributor_id:
    st.subheader("Recettes filtrées selon vos critères")
    st.dataframe(filtered_df[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']].head(10))
