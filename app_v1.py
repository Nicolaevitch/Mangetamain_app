import pandas as pd
import streamlit as st

# Charger le dataset
merged_clean_df = pd.read_csv('base_light_V3.csv')

# Titre de l'application
st.title("Recherche des meilleures recettes par ID")

# Champ d'entrée pour l'ID
user_id = st.text_input("Entrez un ID :", "")

# Vérifier si un ID est saisi
if user_id:
    # Filtrer les données pour l'ID saisi
    filtered_data = merged_clean_df[merged_clean_df['id'] == int(user_id)]
    
    if not filtered_data.empty:
        # Trier par les meilleures notes
        top_recipes = filtered_data.sort_values(by='average_rating', ascending=False).head(10)

        # Sélectionner les colonnes demandées
        top_recipes = top_recipes[['name', 'average_rating', 'minutes', 'n_ingredients']]
        
        # Afficher les résultats
        st.subheader("Top 10 des meilleures recettes")
        st.dataframe(top_recipes)

    else:
        st.warning("Aucune recette trouvée pour cet ID.")
else:
    st.info("Veuillez entrer un ID pour afficher les résultats.")
