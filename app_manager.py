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
import plotly.express as px
import plotly.graph_objects as go


class AppManager:

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
            response = requests.get(image_url,timeout=10)
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


    def perform_tsne_prev(self,recipes : pd.DataFrame, selected_ingredients, contributor_id, n_components=2, n_iter=250):
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
            
            st.write("Vérification des données avant t-SNE :")
            st.write(f"Nombre de recettes après filtrage : {len(filtered_recipes)}")
            st.write(f"Taille des données TF-IDF : {X_tfidf.shape}")

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
            

    def perform_tsne_with_streamlit(self,recipes: pd.DataFrame, selected_ingredients, contributor_id, n_components=2, n_iter=250):
        try:
            # Vérifier si les recettes sont disponibles
            if recipes.empty:
                st.warning("Aucune recette ne correspond aux ingrédients sélectionnés.")
                return

            recipes['dominant_ingredient'] = recipes['ingredients'].apply(
                lambda x: self.get_dominant_ingredient(x, selected_ingredients)
            )

            # Filtrer et vectoriser les ingrédients
            recipes['filtered_ingredients'] = recipes['ingredients'].apply(
                lambda ingredient_list: [ingredient for ingredient in ingredient_list if ingredient in selected_ingredients]
            )
            recipes['filtered_ingredients'] = recipes['filtered_ingredients'].apply(lambda x: ' '.join(x))

            if recipes['filtered_ingredients'].str.strip().eq('').all():
                st.warning("Aucune recette ne contient les ingrédients sélectionnés.")
                return

            vectorizer = TfidfVectorizer(
                tokenizer=lambda x: x.split(),
                stop_words='english',
                max_features=1000
            )
            X_tfidf = vectorizer.fit_transform(recipes['filtered_ingredients'])

            if X_tfidf.shape[0] < 2:
                st.warning("Pas assez de recettes pour appliquer t-SNE.")
                return

            # Appliquer t-SNE
            tsne = TSNE(
                n_components=n_components,
                random_state=42,
                perplexity=30,
                learning_rate=200,
                n_iter=n_iter,
                init='random'
            )
            X_tsne = tsne.fit_transform(X_tfidf.toarray())

            # Ajouter les résultats t-SNE au DataFrame
            recipes['tsne1'] = X_tsne[:, 0]
            recipes['tsne2'] = X_tsne[:, 1]

            # Identifier les points les plus éloignés
            distances = np.linalg.norm(X_tsne[:, np.newaxis] - X_tsne, axis=2)
            np.fill_diagonal(distances, 0)
            max_dist_indices = np.unravel_index(np.argmax(distances), distances.shape)
            recipe_1 = recipes.iloc[max_dist_indices[0]]
            recipe_2 = recipes.iloc[max_dist_indices[1]]

            recipes['highlight'] = ''
            recipes.loc[max_dist_indices[0], 'highlight'] = 'Farthest Recipe 1 ⭐'
            recipes.loc[max_dist_indices[1], 'highlight'] = 'Farthest Recipe 2 ⭐'

            # Créer le graphique interactif avec Plotly pour les dominant_ingredients uniquement
            fig = px.scatter(
                recipes,
                x='tsne1',
                y='tsne2',
                color='dominant_ingredient',
                hover_data=['name', 'id'],
                title='t-SNE Visualization of Recipes',
                color_discrete_sequence=px.colors.qualitative.T10
            )

            # Ajouter les annotations pour les recettes les plus éloignées
            fig.add_trace(
                go.Scatter(
                    x=[recipe_1['tsne1'], recipe_2['tsne1']],
                    y=[recipe_1['tsne2'], recipe_2['tsne2']],
                    mode='markers+text',
                    text=[f"Farthest Recipe 1: {recipe_1['name']}", f"Farthest Recipe 2: {recipe_2['name']}"],
                    textposition=["top right", "top right"],
                    marker=dict(size=12, color='gold', symbol='star'),
                    showlegend=False  # Pas de légende pour les annotations des points éloignés
                )
            )

            # Ajuster l'apparence des axes et de la grille
                # Mettre à jour la mise en page pour cacher les labels des axes et centrer le titre
            fig.update_layout(
                xaxis=dict(
                    title="t-SNE Component 1", 
                    showticklabels=False,  # Cacher les labels sur l'axe X
                    showgrid=True, 
                    zeroline=False
                ),
                yaxis=dict(
                    title="t-SNE Component 2", 
                    showticklabels=False,  # Cacher les labels sur l'axe Y
                    showgrid=True, 
                    zeroline=False
                ),
                legend_title="Ingrédient Dominant",  # Titre de la légende
                title=dict(
                    text="Visualisation des recettes par la t-SNE",  # Texte du titre
                    font=dict(size=16),
                    x=0.5,  # Centrer le titre horizontalement
                    xanchor='center'  # Ancrage du titre centré
                ),
            )

            # Afficher le graphique dans Streamlit
            st.plotly_chart(fig, use_container_width=True)
            
            
            # Ajouter un espacement de 4 lignes
            st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
            
            st.markdown(
            "**t-SNE (t-Distributed Stochastic Neighbor Embedding)** est un algorithme de réduction de dimensionnalité non linéaire. Il est particulièrement efficace pour projeter des données à haute dimension dans un espace à faible dimension, tout en préservant les structures locales et globales des données.L'objectif principal de t-SNE est de représenter chaque objet des données d'origine par un point dans un espace de plus faible dimension, de manière à ce que les objets similaires soient représentés par des points proches et les objets dissemblables soient représentés par des points éloignés avec une forte probabilité.")
            st.markdown("Cette technique est largement utilisée pour la visualisation de données complexes, telles que des jeux de données textuels, des images ou des vecteurs d'embedding. Elle repose sur la modélisation des relations de voisinage entre les points, en utilisant une mesure de similarité basée sur des distributions de probabilité, et optimise ces relations dans l'espace projeté.")

            images = ["https://i.imghippo.com/files/NWt4811myk.PNG",
                      "https://i.imghippo.com/files/pId6918aM.PNG",                      
                      "https://i.imghippo.com/files/Dq5421IAg.PNG",
                      "https://i.imghippo.com/files/INX9117lTQ.PNG", 
                      "https://i.imghippo.com/files/xWxW1858tEI.PNG"]  # Remplacez par vos propres images

            # Affichage des images avec un espacement
            text_1 = r'''
                            \text{La probabilité gaussienne qu'une observation } x_{i} \text{ soit voisine de } x_{j} \text{est définie par }
                            '''
            text_2 = r'''
                            \text{La matrice des probabilités jointes contient les termes } p_{ij} \text{suivants :}  
                            '''
            text_3 = r'''
                            \text{Ensuite, cette formule ci-dessous représente la proximité entre nos données dans un espace en faible dimension, notées } \{ \mathbf{y}_i \}_{i=1}^n
                      '''
            text_4 = r'''
                            \text{L'algorithme final aura pour but de minimiser le critère suivant :} 
                            '''
            texts = [text_1,text_2,text_3,text_4]
            
            st.markdown("<br>" * 1,unsafe_allow_html=True)            
            
            st.markdown("*Voici quelques équations décrivant cet algorithme*",unsafe_allow_html=True)  
            
            for image,text in zip(images,texts):
                st.latex(text)
                st.image(image)  # Affiche l'image
                st.markdown("<br>" * 1,unsafe_allow_html=True)            
            
        except Exception as e:
            st.error(f"Erreur lors de la génération du graphique t-SNE : {e}")
                
    # Identifier l'ingrédient dominant
    def get_dominant_ingredient(self, ingredient_list, selected_ingredients):
        for ingredient in selected_ingredients:
            if ingredient in ingredient_list:
                return ingredient
        return 'Other'


