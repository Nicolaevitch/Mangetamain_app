import pandas as pd
import streamlit as st

# Charger le dataset
merged_clean_df = pd.read_csv('base_light_V3.csv')

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

# Champ d'entrée pour l'ID
user_id = st.text_input("Entrez un ID :", "")

# Vérifier si un ID est saisi
if user_id:
    try:
        # Convertir l'ID en entier
        user_id = int(user_id)
        
        # Filtrer les données pour l'utilisateur
        filtered_data = filtered_df[filtered_df['id'] == user_id]
        
        if not filtered_data.empty:
            # Calculer le total des recettes pour cet utilisateur
            total_recipes = filtered_df[filtered_df['id'] == user_id].shape[0]

            # Trier par les meilleures notes et sélectionner les colonnes importantes
            top_recipes = filtered_data.sort_values(by='average_rating', ascending=False).head(10)
            top_recipes = top_recipes[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']]
            
            # Afficher les statistiques de l'utilisateur
            st.subheader(f"Statistiques pour l'utilisateur avec l'ID {user_id}")
            st.write(f"**Total de recettes publiées :** {total_recipes}")

            # Afficher les meilleures recettes de l'utilisateur
            st.subheader(f"Top 10 des meilleures recettes pour l'ID {user_id}")
            st.dataframe(top_recipes)
        else:
            st.warning("Aucune recette trouvée pour cet ID.")
    except ValueError:
        st.error("Veuillez entrer un ID valide.")
else:
    st.info("Veuillez entrer un ID pour afficher les résultats.")

# Afficher les données filtrées avec les filtres interactifs
if not user_id:
    st.subheader("Recettes filtrées selon vos critères")
    st.dataframe(filtered_df[['name', 'average_rating', 'minutes', 'palmarès', 'steps_category']].head(10))
