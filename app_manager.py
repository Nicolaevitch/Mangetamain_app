import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import requests
import io
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List
from PIL import Image

class AppManager():

    def hide_streamlit_ui_elements(self, hide_menu: bool = True, hide_footer: bool = True, custom_class: str = None):
        """
        Masque certains éléments de l'interface Streamlit, comme le menu, le footer, ou des éléments spécifiques.

        Parameters:
        ----------
        hide_menu : bool, optional
            Indique si le menu de Streamlit doit être masqué. Par défaut, True.
        hide_footer : bool, optional
            Indique si le footer de Streamlit doit être masqué. Par défaut, True.
        custom_class : str, optional
            Nom de classe CSS spécifique à masquer. Si None, aucun div supplémentaire ne sera masqué.

        Returns:
        -------
        None
            Applique les styles CSS pour masquer les éléments dans l'interface utilisateur Streamlit.

        Raises:
        ------
        Exception
            Si une erreur survient lors de l'application des styles.
        """
        try:
            # Construire le style CSS de manière dynamique
            hide_streamlit_style = """
            <style>
            """
            if hide_menu:
                hide_streamlit_style += """
                header {visibility: hidden;}
                """
            if hide_footer:
                hide_streamlit_style += """
                .streamlit-footer {display: none;}
                """
            if custom_class:
                hide_streamlit_style += f"""
                .{custom_class} {{display: none;}}
                """
            hide_streamlit_style += """
            </style>
            """
            # Appliquer le style CSS
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur lors du masquage des éléments Streamlit : {e}")

    def suggest_similar_ids(self,table_recipes : pd.DataFrame, user_input: str, max_suggestions: int = 3) -> List[int]:
        """
        Propose des IDs similaires basés sur la distance de Jaccard entre l'entrée utilisateur
        et les contributeurs dans la table des recettes.

        Parameters:
        ----------
        table_recipes : pd.DataFrame
            Dataframe des recettes
        user_input : str
            Identifiant de l'utilisateur sous forme de chaîne.
        max_suggestions : int, optional
            Nombre maximum d'IDs similaires à retourner (par défaut 3).
        

        Returns:
        -------
        List[int]
            Liste des IDs similaires triés par distance croissante.
        """
        try:
            user_id_set = set(user_input)
            distances = table_recipes["contributor_id"].apply(
                lambda x: self.jaccard_similarity(user_id_set, set(str(x)))
            )
            table_recipes["jaccard_distance"] = distances
            closest_ids = table_recipes.nsmallest(max_suggestions, "jaccard_distance")["contributor_id"]
            return closest_ids.tolist()
        except Exception as e:
            logger.error(f"Erreur dans la suggestion d'IDs : {e}")
            return []

    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """
        Calcule la distance de Jaccard entre deux ensembles.

        Parameters:
        ----------
        set1 : set
            Premier ensemble à comparer.
        set2 : set
            Second ensemble à comparer.

        Returns:
        -------
        float
            Distance de Jaccard (1 - similarité de Jaccard). Retourne `float('inf')` en cas d'erreur.
        """
        try:
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return 1 - (intersection / union)
        except Exception as e:
            logger.error(f"Erreur dans le calcul de la similarité de Jaccard : {e}")
            return float('inf')

    def set_image(self, image_url: str):
        """
        Télécharge une image depuis une URL et l'affiche dans l'application.

        Parameters:
        ----------
        url_im : str
            URL de l'image à télécharger et afficher.

        Returns:
        -------
        None
            Affiche l'image dans l'interface utilisateur Streamlit.

        Raises:
        ------
        requests.RequestException
            En cas d'erreur de téléchargement de l'image depuis l'URL.
        Exception
            Pour toute autre erreur non prévue.
        """
        try:
            # Télécharger l'image distante
            response = requests.get(url_im,timeout=10)
            response.raise_for_status()  # Vérifie les erreurs HTTP

            # Lire l'image directement depuis les données téléchargées
            image = Image.open(io.BytesIO(response.content))
            image = image.resize((200, 100))

            # Afficher l'image
            st.image(image, caption="", use_container_width=True)

        except requests.RequestException as e:
            st.error(f"Erreur lors du téléchargement de l'image : {e}")
        except Exception as e:
            st.error(f"Une erreur est survenue : {e}")
            
            
    def set_background_image(self, image_url: str):
        """
        Définit une image de fond pour l'application Streamlit à partir d'une URL.

        Parameters:
        ----------
        image_url : str
            URL de l'image à utiliser comme arrière-plan.

        Returns:
        -------
        None
            Applique l'image de fond dans l'interface utilisateur Streamlit.

        Raises:
        ------
        Exception
            Si une erreur survient lors de l'application de l'image.
        """
        try:
            # Générer le style CSS pour l'image de fond
            background_style = f"""
            <style>
            .stApp {{
                background: url("{image_url}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """
            # Appliquer le style
            st.markdown(background_style, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur lors de l'application de l'image de fond : {e}")
            

    def display_adjusted_title(self, title: str, margin_top: int = 50, emoji: str = "🍲"):
        """
        Affiche un titre personnalisé avec un style ajusté pour Streamlit.

        Parameters:
        ----------
        title : str
            Le texte du titre à afficher.
        margin_top : int, optional
            La marge supérieure du titre (en pixels). Par défaut, 50.
        emoji : str, optional
            Un emoji à afficher à côté du titre. Par défaut, "🍲".

        Returns:
        -------
        None
            Le titre est directement rendu dans l'interface utilisateur Streamlit.
        """
        try:
            # Créer le style CSS pour ajuster la position du titre
            adjust_title_style = f"""
            <style>
            h1 {{
                position: fixed;
                margin-top: {margin_top}px; /* Ajuste la marge supérieure */
            }}
            </style>
            """
            # Appliquer le style et afficher le titre
            st.markdown(adjust_title_style, unsafe_allow_html=True)
            st.markdown(f'<h1 class="title">{title} {emoji}</h1>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Erreur lors de l'affichage du titre ajusté : {e}")


    def perform_tsne(recipes, selected_ingredients, contributor_id, n_components=2, n_iter=250):
        """
        Effectue une réduction dimensionnelle t-SNE directement sur les données vectorisées sans PCA préalable,
        sans échantillonner les données, et en filtrant également les recettes par contributor_id et par les ingredients que l'utilisateur choisit.
        Puis génère une visualisation t-SNE des recettes basées sur les ingrédients sélectionnés, 
        et affiche les deux recettes les plus éloignées sur le graphique avec leurs noms. Les résultats sont affichés 
        dans une interface Streamlit si utilisée.

        Args:
            recipes (pd.DataFrame): 
                Un DataFrame contenant les recettes. Chaque recette doit avoir les colonnes suivantes :
                - 'ingredients' : Liste des ingrédients sous forme de texte ou de chaîne de caractères.
                - 'contributor_id' : Identifiant unique du contributeur.
                - 'name' : Nom de la recette.
            
            selected_ingredients (list of str): 
                Liste d'ingrédients utilisés pour filtrer les recettes.

            contributor_id (str or int): 
                Identifiant unique du contributeur pour filtrer les recettes spécifiquement pour cet utilisateur.
            
            n_components (int, optionnel, par défaut=2): 
                Nombre de dimensions de sortie pour la réduction t-SNE. Par défaut, 2 pour une visualisation en 2D.
            
            n_iter (int, optionnel, par défaut=250): 
                Nombre d'itérations que t-SNE effectuera lors de la réduction de dimension. Plus ce nombre est élevé, 
                plus la convergence sera précise, mais cela prendra aussi plus de temps de calcul.

        Returns:
            None : 
                Cette fonction génère une visualisation t-SNE des recettes filtrées et affiche les noms des deux recettes 
                les plus éloignées sur le graphique. Si un problème survient (par exemple, un nombre insuffisant de recettes), 
                un avertissement sera affiché dans l'interface Streamlit.
        
        Exceptions :
            Si des erreurs surviennent lors de la génération du graphique ou du traitement des données, un avertissement 
            sera affiché dans l'interface Streamlit.
        """
        
        try:
            # Filtrer les recettes basées sur les ingrédients sélectionnés
            recipes['filtered_ingredients'] = recipes['ingredients'].apply(
                lambda x: ' '.join([word for word in x.split() if word in selected_ingredients])
            )

            # Filtrer les recettes pour le contributeur spécifié (contributor_id)
            filtered_recipes = recipes[recipes['contributor_id'] == contributor_id]  # Filtrer par contributor_id

            # Filtrer les recettes qui ont des ingrédients valides après le filtrage
            filtered_recipes = filtered_recipes[filtered_recipes['filtered_ingredients'] != '']

            if filtered_recipes.empty:
                print("Aucune recette après filtrage, essayez avec d'autres ingrédients.")
                return

            # Identifier un ingrédient dominant
            def get_dominant_ingredient(ingredients):
                ingredient_list = ingredients.split() if ingredients else []
                for ingredient in selected_ingredients:
                    if ingredient in ingredient_list:
                        return ingredient
                return 'Other'

            # Vérification que la colonne filtered_ingredients n'est pas vide
            if filtered_recipes['filtered_ingredients'].isnull().any():
                print("Certaines recettes ont des ingrédients filtrés vides.")
                return

            # Appliquer la fonction de l'ingrédient dominant
            filtered_recipes['dominant_ingredient'] = filtered_recipes['filtered_ingredients'].apply(get_dominant_ingredient)

            # Vectorisation avec TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)  # Limiter le nombre de features
            X_tfidf = vectorizer.fit_transform(filtered_recipes['filtered_ingredients'])

            # Vérification de la taille des données après la vectorisation
            if X_tfidf.shape[0] < 50:
                print("Il n'y a pas assez de recettes pour générer une visualisation.")
                return

            # Réduction dimensionnelle avec t-SNE
            tsne = TSNE(n_components=n_components, random_state=42, perplexity=30, learning_rate=200, n_jobs=-1, max_iter=n_iter, init='random')
            X_tsne = tsne.fit_transform(X_tfidf)  # pas de toarray() ici

            # Ajouter les coordonnées t-SNE au dataset
            filtered_recipes['tsne1'] = X_tsne[:, 0]
            filtered_recipes['tsne2'] = X_tsne[:, 1]

            # Calculer la distance euclidienne entre chaque paire de points
            distances = np.linalg.norm(X_tsne[:, np.newaxis] - X_tsne, axis=2)

            # Trouver les indices des deux points les plus éloignés
            np.fill_diagonal(distances, 0)  # Ignorer la diagonale (distance d'un point à lui-même)
            max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)

            # Récupérer les indices des recettes correspondantes
            point_1_index, point_2_index = max_dist_indices
            recipe_1 = filtered_recipes.iloc[point_1_index]
            recipe_2 = filtered_recipes.iloc[point_2_index]

            # Afficher les recettes les plus éloignées
            print(f"Recette 1: {recipe_1['name']}, Dominant Ingredient: {recipe_1['dominant_ingredient']}")
            print(f"Recette 2: {recipe_2['name']}, Dominant Ingredient: {recipe_2['dominant_ingredient']}")

            # Visualisation avec Seaborn
            plt.figure(figsize=(12, 8))
            scatter = sns.scatterplot(
                x='tsne1',
                y='tsne2',
                hue='dominant_ingredient',
                data=filtered_recipes,
                palette="tab20",
                s=100,
                marker='o'
            )
            plt.title("t-SNE Visualization of Recipe Ingredients (Dominant Ingredients)", fontsize=16)
            plt.xlabel("t-SNE Component 1", fontsize=12)
            plt.ylabel("t-SNE Component 2", fontsize=12)
            plt.legend(title="Dominant Ingredient", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks([])  # Supprimer les ticks sur l'axe X
            plt.yticks([])  # Supprimer les ticks sur l'axe Y
            plt.grid(True, linestyle='--', alpha=0.5)
            

            # Annoter les deux points les plus éloignés avec les noms des recettes
            scatter.annotate(
                recipe_1['name'],
                xy=(recipe_1['tsne1'], recipe_1['tsne2']),
                xytext=(recipe_1['tsne1'] + 0.5, recipe_1['tsne2'] + 0.5),
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                fontsize=10,
                color='black'
            )
            scatter.annotate(
                recipe_2['name'],
                xy=(recipe_2['tsne1'], recipe_2['tsne2']),
                xytext=(recipe_2['tsne1'] + 0.5, recipe_2['tsne2'] - 0.5),
                arrowprops=dict(facecolor='black', arrowstyle="->"),
                fontsize=10,
                color='black'
            )
            
            st.pyplot(plt)


            # Afficher le graphique
            plt.show()

        except Exception as e:
            print(f"Erreur lors de la génération du graphique t-SNE : {e}")

# Exemple d'appel
#app_manager = AppManager()
#recipes = pd.read_csv("pp_recipes_cleaned.csv")
#contributor_id = 133174
#selected_ingredients = ['salt', 'pepper']
#app_manager.perform_tsne(recipes, selected_ingredients, contributor_id)
