import pandas as pd
from src.FindingCloseRecipes.recipe_finder import RecipeFinder

def run_recipe_finder(recipe_id):
    """
    Trouve les 100 recettes les plus proches d'une recette donnée par son ID.
    
    Args:
        recipe_id (int): L'identifiant de la recette pour laquelle chercher les recettes similaires.
    
    Returns:
        pd.DataFrame: Les 100 recettes les plus proches avec leurs distances combinées.
    """
    # Charger le dataset
    pp_recipes = pd.read_csv('data/pp_recipes.csv')  # Remplacez par le chemin de votre fichier

    # Initialiser le RecipeFinder
    finder = RecipeFinder(pp_recipes)
    finder.preprocess()  # Cette étape inclut désormais des matrices creuses

    # Trouver les recettes similaires
    try:
        similar_recipes = finder.find_similar_recipes(recipe_id)
        
        # Affichage pour vérification
        print(f"Recette {recipe_id}:")
        print(pp_recipes[pp_recipes['id'] == recipe_id])
        print("100 plus proches recettes:")
        print(similar_recipes)
        
        return similar_recipes
    except ValueError as e:
        print(e)
        return pd.DataFrame()

# Exemple d'utilisation
if __name__ == "__main__":
    recipe_id_input = 137739  # ID de la recette à rechercher (modifiez ici ou passez en argument)
    run_recipe_finder(recipe_id_input)


