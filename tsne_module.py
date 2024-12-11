import logging
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import streamlit as st
from typing import List


# Configurer le logging
def setup_logging():
    # Créer un logger pour les événements généraux
    logger = logging.getLogger('recipes_tsne')
    logger.setLevel(logging.DEBUG)

    # Créer un handler pour enregistrer les événements dans un fichier de log
    handler = logging.FileHandler('tsne_debug.log')
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Créer un handler pour enregistrer les erreurs dans un autre fichier
    error_handler = logging.FileHandler('tsne_error.log')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(error_handler)

    return logger


class RecipeTSNE:
    """
    Classe pour effectuer la réduction de dimensionnalité t-SNE sur un ensemble de données de recettes.

    Cette classe est responsable de l'application de l'algorithme t-SNE (t-Distributed Stochastic Neighbor Embedding)
    sur un jeu de données contenant des recettes. Elle filtre les recettes en fonction des ingrédients sélectionnés,
    utilise TF-IDF pour les vectoriser, puis applique t-SNE pour réduire la dimensionnalité et afficher les résultats
    à l'aide de Streamlit et Plotly.

    Attributs:
        logger (logging.Logger): Un objet logger pour l'enregistrement des événements et erreurs.
    """
    def __init__(self):
        self.logger = setup_logging()

    def perform_tsne_with_streamlit(
        self,
        recipes: pd.DataFrame,
        selected_ingredients: List[str],
        contributor_id: int,
        n_components: int = 2,
        n_iter: int = 250
    ) -> None:
        """
        Applique la réduction de dimensionnalité t-SNE sur un ensemble de données de recettes et visualise les résultats avec Streamlit.

        Cette fonction filtre les recettes fournies en fonction des ingrédients sélectionnés, les vectorise à l'aide de TF-IDF,
        et applique l'algorithme t-SNE pour réduire la dimensionnalité. Elle crée ensuite un graphique interactif
        des données réduites avec Plotly, en mettant en évidence les deux recettes les plus éloignées.

        Args:
            recipes (pd.DataFrame): Un DataFrame contenant les données des recettes. Doit inclure une colonne `ingredients`.
            selected_ingredients (list): Une liste des ingrédients à utiliser pour filtrer les recettes.
            contributor_id (int): Identifiant du contributeur (non utilisé dans cette implémentation).
            n_components (int, optionnel): Le nombre de composantes pour le t-SNE. Par défaut à 2.
            n_iter (int, optionnel): Le nombre d'itérations pour l'optimisation du t-SNE. Par défaut à 250.

        Raises:
            Exception: En cas d'erreur lors du traitement ou de la visualisation.

        Returns:
            None: La visualisation est affichée dans l'application Streamlit.
        """
        try:
            if recipes.empty:
                self.logger.warning("Aucune recette ne correspond aux ingrédients sélectionnés.")
                st.warning("Aucune recette ne correspond aux ingrédients sélectionnés.")
                return

            recipes['dominant_ingredient'] = recipes['ingredients'].apply(
                lambda x: self.get_dominant_ingredient(x, selected_ingredients)
            )

            recipes['filtered_ingredients'] = recipes['ingredients'].apply(
                lambda ingredient_list: [ingredient for ingredient in ingredient_list if ingredient in selected_ingredients]
            )
            recipes['filtered_ingredients'] = recipes['filtered_ingredients'].apply(lambda x: ' '.join(x))

            if recipes['filtered_ingredients'].str.strip().eq('').all():
                self.logger.warning("Aucune recette ne contient les ingrédients sélectionnés.")
                st.warning("Aucune recette ne contient les ingrédients sélectionnés.")
                return

            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                stop_words='english',
                max_features=1000
            )
            X_tfidf = vectorizer.fit_transform(recipes['filtered_ingredients'])

            if X_tfidf.shape[0] < 2:
                self.logger.warning("Pas assez de recettes pour appliquer t-SNE.")
                st.warning("Pas assez de recettes pour appliquer t-SNE.")
                return

            tsne = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=30,
                learning_rate=200,
                n_iter=n_iter,
                init='random'
            )
            X_tsne = tsne.fit_transform(X_tfidf.toarray())

            recipes['tsne1'] = X_tsne[:, 0]
            recipes['tsne2'] = X_tsne[:, 1]

            distances = np.linalg.norm(X_tsne[:, np.newaxis] - X_tsne, axis=2)
            np.fill_diagonal(distances, 0)
            max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)
            recipe_1 = recipes.iloc[max_dist_indices[0]]
            recipe_2 = recipes.iloc[max_dist_indices[1]]

            recipes['highlight'] = ''
            recipes.loc[max_dist_indices[0], 'highlight'] = 'Farthest Recipe 1 ⭐'
            recipes.loc[max_dist_indices[1], 'highlight'] = 'Farthest Recipe 2 ⭐'

            fig = px.scatter(
                recipes,
                x='tsne1',
                y='tsne2',
                color='dominant_ingredient',
                hover_data=['name', 'id'],
                title='t-SNE Visualization of Recipes',
                color_discrete_sequence=px.colors.qualitative.T10
            )

            fig.add_trace(
                go.Scatter(
                    x=[recipe_1['tsne1'], recipe_2['tsne1']],
                    y=[recipe_1['tsne2'], recipe_2['tsne2']],
                    mode='markers+text',
                    text=[f"Farthest Recipe 1: {recipe_1['name']}", f"Farthest Recipe 2: {recipe_2['name']}"],
                    textposition=["top right", "top right"],
                    marker=dict(size=12, color='gold', symbol='star'),
                    showlegend=False
                )
            )

            fig.update_layout(
                xaxis=dict(
                    title="t-SNE Component 1", 
                    showticklabels=False,  
                    showgrid=True, 
                    zeroline=False
                ),
                yaxis=dict(
                    title="t-SNE Component 2", 
                    showticklabels=False,  
                    showgrid=True, 
                    zeroline=False
                ),
                legend_title="Dominant Ingredient",
                title=dict(
                    text="t-SNE Visualization of Recipes",
                    font=dict(size=16),
                    x=0.5,
                    xanchor='center'
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)

            st.markdown(
                "**Le TSNE est un algorithme permettant de réduire la dimension d’une matrice tout en préservant les informations importantes contenues à l’intérieur. Il s'agit d'une technique non linéaire bien adaptée à l'intégration de données à haute dimension pour la visualisation dans un espace à basse dimension. Elle modélise chaque objet par un point de manière à ce que les objets similaires soient modélisés par des points proches et que les objets dissemblables soient modélisés par des points éloignés avec une probabilité élevée**."
            )

            images = [
                "https://i.imghippo.com/files/NWt4811myk.PNG",
                "https://i.imghippo.com/files/pId6918aM.PNG",
                "https://i.imghippo.com/files/Dq5421IAg.PNG",
                "https://i.imghippo.com/files/INX9117lTQ.PNG",
                "https://i.imghippo.com/files/xWxW1858tEI.PNG"
            ]

            texts = [
                r'''La probabilité gaussienne que \( x_i \) soit voisin de \( x_j \) est définie par''',
                r'''La matrice des probabilités contient donc les termes \( p_{ij} \) suivants :''',
                r'''Ensuite, nous obtenons cette équation ci-dessous qui représente la proximité entre nos données dans un espace en faible dimension, \( y_{i,(1,N)} \) qui désigne nos observations.''',
                r'''Enfin, l'algorithme final aura pour but de minimiser le critère suivant :'''
            ]

            st.markdown("<br>" * 1, unsafe_allow_html=True)
            st.markdown("*Voici quelques équations décrivant cet algorithme*", unsafe_allow_html=True)

            for image, text in zip(images, texts):
                st.latex(text)
                st.image(image)
                st.markdown("<br>" * 1, unsafe_allow_html=True)

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du graphique t-SNE : {e}")
            st.error(f"Erreur lors de la génération du graphique t-SNE : {e}")

    def get_dominant_ingredient(self, ingredient_list: List[str], selected_ingredients: List[str]) -> str:
        """
        Identifie l'ingrédient dominant dans une liste d'ingrédients.

        Cette fonction vérifie si l'un des ingrédients sélectionnés est présent dans la liste d'ingrédients fournie 
        et retourne la première correspondance. Si aucun ingrédient sélectionné n'est trouvé, elle retourne 'Other'.

        Args:
            ingredient_list (List[str]): Une liste d'ingrédients issus d'une recette.
            selected_ingredients (List[str]): Une liste des ingrédients sélectionnés à vérifier.

        Returns:
            str: L'ingrédient dominant s'il est trouvé, ou 'Other' si aucune correspondance n'est trouvée.

        Raises:
            ValueError: Si l'un des arguments n'est pas une liste.
        """
        try:
            # Vérification des types des arguments
            if not isinstance(ingredient_list, list) or not isinstance(selected_ingredients, list):
                raise ValueError("Les arguments doivent être des listes.")

            # Vérifier chaque ingrédient dans la liste sélectionnée
            for ingredient in selected_ingredients:
                if ingredient in ingredient_list:
                    self.logger.debug(f"Ingrédient dominant trouvé: {ingredient}")
                    return ingredient

            self.logger.debug("Aucun ingrédient dominant trouvé, retour de 'Other'.")
            return 'Other'

        except ValueError as e:
            self.logger.error(f"Erreur de valeur dans get_dominant_ingredient: {e}")
            raise  # Relancer l'exception pour qu'elle puisse être gérée ailleurs si nécessaire
        except Exception as e:
            self.logger.critical(f"Erreur inconnue dans get_dominant_ingredient: {e}")
            raise