# recipe_finder.py
import pandas as pd
import numpy as np
from src.FindingCloseRecipes.config import NUMERIC_FEATURES, DEFAULT_WEIGHTS, COMBINED_WEIGHTS, TOP_N
from src.FindingCloseRecipes.distances import DistanceCalculator
from src.FindingCloseRecipes.vectorizers import Vectorizer

class RecipeFinder:
    def __init__(self, recipes_df):
        self.recipes_df = recipes_df
        self.id_to_index = pd.Series(recipes_df.index, index=recipes_df['id'])

    def preprocess(self):
        self.numeric_df = self.recipes_df[NUMERIC_FEATURES]
        self.weights_array = np.array([DEFAULT_WEIGHTS[feature] for feature in NUMERIC_FEATURES])
        
        self.tfidf_name, _ = Vectorizer.tfidf_vectorize(self.recipes_df['name'])
        self.tfidf_tags, _ = Vectorizer.tfidf_vectorize(self.recipes_df['tags'])
        self.tfidf_steps, _ = Vectorizer.tfidf_vectorize(self.recipes_df['steps'])
        self.bow_ingredients, _ = Vectorizer.bow_vectorize(self.recipes_df['ingredients'])

    def find_similar_recipes(self, recipe_id):
        if recipe_id not in self.id_to_index:
            raise ValueError("Identifiant de recette introuvable.")
        
        recipe_index = self.id_to_index[recipe_id]
        
        # Calculer les distances avec les matrices creuses
        distance_name = DistanceCalculator.cosine_distance_sparse(
            recipe_id=recipe_id,
            tfidf_matrix=self.tfidf_name,
            id_to_index=self.id_to_index,
            index_to_id=self.id_to_index.index
        )
        distance_tags = DistanceCalculator.cosine_distance_sparse(
            recipe_id=recipe_id,
            tfidf_matrix=self.tfidf_tags,
            id_to_index=self.id_to_index,
            index_to_id=self.id_to_index.index
        )
        distance_steps = DistanceCalculator.cosine_distance_sparse(
            recipe_id=recipe_id,
            tfidf_matrix=self.tfidf_steps,
            id_to_index=self.id_to_index,
            index_to_id=self.id_to_index.index
        )
        distance_ingredients = DistanceCalculator.cosine_distance_sparse(
            recipe_id=recipe_id,
            tfidf_matrix=self.bow_ingredients,
            id_to_index=self.id_to_index,
            index_to_id=self.id_to_index.index
        )
        
        # Calculer les distances pour les variables numériques
        distance_numeric = DistanceCalculator.euclidean_distance(
            self.numeric_df, recipe_index, self.weights_array
        )

        # Combiner les distances avec les poids
        combined_distance = (
            COMBINED_WEIGHTS["alpha"] * distance_name +
            COMBINED_WEIGHTS["beta"] * distance_tags +
            COMBINED_WEIGHTS["gamma"] * distance_steps +
            COMBINED_WEIGHTS["delta"] * distance_ingredients +
            COMBINED_WEIGHTS["epsilon"] * distance_numeric
        )
        
        # Trier les indices par distance combinée croissante
        sorted_indices = np.argsort(combined_distance)
        sorted_indices = sorted_indices[sorted_indices != recipe_index]
        
        # Récupérer les recettes les plus proches
        top_n_indices = sorted_indices[:TOP_N]
        similar_recipes = self.recipes_df.iloc[top_n_indices].copy()
        similar_recipes['combined_distance'] = combined_distance[top_n_indices]
        
        return similar_recipes



