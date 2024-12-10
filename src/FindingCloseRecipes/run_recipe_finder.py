from src.FindingCloseRecipes.vectorizers import RecipeVectorizer
from src.FindingCloseRecipes.distances import RecipeDistanceCalculator
from src.FindingCloseRecipes.recipe_finder import RecipeFinder
from src.FindingCloseRecipes.config import NUMERIC_FEATURES, WEIGHTS, WEIGHTS_NUM

import pandas as pd
import numpy as np

import pandas as pd

def reconstruct_pp_recipes():
    # Chargement et concaténation des fichiers par colonnes
    tags = pd.concat([
        pd.read_csv(f"data/pp_recipes_tags_{i}.csv") for i in range(1, 5)
    ])
    name = pd.concat([
        pd.read_csv(f"data/pp_recipes_name_{i}.csv") for i in range(1, 5)
    ])
    steps = pd.concat([
        pd.read_csv(f"data/pp_recipes_steps_{i}.csv") for i in range(1, 5)
    ])
    ingredients = pd.concat([
        pd.read_csv(f"data/pp_recipes_ingredients_{i}.csv") for i in range(1, 5)
    ])
    numerics = pd.concat([
        pd.read_csv(f"data/pp_recipes_numerics_{i}.csv") for i in range(1, 5)
    ])

    # Fusion des datasets selon la colonne "id"
    pp_recipes = tags.merge(name, on="id", how="inner")
    pp_recipes = pp_recipes.merge(steps, on="id", how="inner")
    pp_recipes = pp_recipes.merge(ingredients, on="id", how="inner")
    pp_recipes = pp_recipes.merge(numerics, on="id", how="inner")

    return pp_recipes

def run_recipe_finder(recipe_id):
    """
    Exécute le pipeline complet pour trouver les indices des recettes les plus proches.
    
    :param recipe_id: int, Identifiant de la recette de référence
    :return: list, Indices des recettes les plus proches
    """
    # Charger les données
    pp_recipes = reconstruct_pp_recipes()

    # Étape 1 : Vectorisation des colonnes textuelles

    vectorizer = RecipeVectorizer(pp_recipes)
    tfidf_name = vectorizer.vectorize_tfidf('name')
    tfidf_tags = vectorizer.vectorize_tfidf('tags')
    tfidf_steps = vectorizer.vectorize_tfidf('steps')
    bow_ingredients = vectorizer.vectorize_bow('ingredients')

    # Étape 2 : Calcul des distances pour chaque type de données
    calculator = RecipeDistanceCalculator(pp_recipes)
    distance_name = calculator.distance_for_recipe_tfidf(recipe_id, tfidf_name)
    distance_tags = calculator.distance_for_recipe_tfidf(recipe_id, tfidf_tags)
    distance_steps = calculator.distance_for_recipe_tfidf(recipe_id, tfidf_steps)
    distance_ingredients = calculator.distance_for_recipe_bow(recipe_id, bow_ingredients)

    # Distance combinée
    weights_array = np.array([WEIGHTS_NUM[feature] for feature in NUMERIC_FEATURES])
    numeric_df = pp_recipes[NUMERIC_FEATURES]
    distance_numeric = calculator.distance_euclidean_for_recipe(recipe_id, numeric_df, weights_array)

    # Étape 3 : Combinaison des distances et recherche des indices des recettes les plus proches
    distances_dict = {
        'name': distance_name,
        'tags': distance_tags,
        'steps': distance_steps,
        'ingredients': distance_ingredients,
        'numeric': distance_numeric
    }

    finder = RecipeFinder(pp_recipes)
    closest_indices = finder.find_closest_recipes(recipe_id, distances_dict, WEIGHTS, top_n=100)

    # Retourner les indices pour d'autres utilisations
    return closest_indices

