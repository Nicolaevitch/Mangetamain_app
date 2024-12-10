import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class DistanceCalculator:
    @staticmethod
    def euclidean_distance(numeric_df, recipe_index, weights_array):
        """
        Calcule la distance euclidienne pondérée pour des données numériques.
        """
        recipe_vector = numeric_df.iloc[recipe_index].values
        differences = numeric_df.values - recipe_vector
        squared_diff = differences ** 2
        weighted_squared_diff = squared_diff * weights_array
        return np.sqrt(np.sum(weighted_squared_diff, axis=1))

    @staticmethod
    def cosine_distance_sparse(recipe_id, tfidf_matrix, id_to_index, index_to_id):
        """
        Calcule la distance cosinus pour une matrice creuse (sparse).
        """
        # Vérifier si l'identifiant existe
        if recipe_id not in id_to_index:
            raise ValueError("Identifiant de recette introuvable.")

        # Obtenir l'index de la recette dans la matrice
        recipe_idx = id_to_index[recipe_id]

        # Extraire le vecteur TF-IDF de la recette
        recipe_vector = tfidf_matrix[recipe_idx]

        # Calculer les similarités cosinus
        cosine_similarities = cosine_similarity(recipe_vector, tfidf_matrix).flatten()

        # Calculer les distances
        distances = 1 - cosine_similarities

        return distances
    
    

