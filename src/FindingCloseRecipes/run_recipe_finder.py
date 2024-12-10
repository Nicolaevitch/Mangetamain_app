import pandas as pd
from src.FindingCloseRecipes.recipe_finder import RecipeFinder

def reconstruct_pp_recipes():
    datasets = {}
    # Colonnes et fichiers à charger
    columns = ["tags", "name", "steps", "ingredients", "numerics"]
    for column in columns:
        datasets[column] = []
        for i in range(1, 5):
            file_path = f"data/pp_recipes_{column}_{i}.csv"
            if os.path.exists(file_path):
                datasets[column].append(pd.read_csv(file_path))
            else:
                print(f"Fichier manquant : {file_path}")
    
    # Concaténation des fichiers pour chaque colonne
    tags = pd.concat(datasets["tags"], ignore_index=True) if datasets["tags"] else pd.DataFrame()
    name = pd.concat(datasets["name"], ignore_index=True) if datasets["name"] else pd.DataFrame()
    steps = pd.concat(datasets["steps"], ignore_index=True) if datasets["steps"] else pd.DataFrame()
    ingredients = pd.concat(datasets["ingredients"], ignore_index=True) if datasets["ingredients"] else pd.DataFrame()
    numerics = pd.concat(datasets["numerics"], ignore_index=True) if datasets["numerics"] else pd.DataFrame()

    # Fusion des datasets selon la colonne "id"
    if not tags.empty:
        pp_recipes = tags
    for df in [name, steps, ingredients, numerics]:
        if not df.empty:
            pp_recipes = pp_recipes.merge(df, on="id", how="inner")

    return pp_recipes

def run_recipe_finder(recipe_id):
    """
    Trouve les 100 recettes les plus proches d'une recette donnée par son ID.
    
    Args:
        recipe_id (int): L'identifiant de la recette pour laquelle chercher les recettes similaires.
    
    Returns:
        pd.DataFrame: Les 100 recettes les plus proches avec leurs distances combinées.
    """
    # Charger le dataset
    pp_recipes = reconstruct_pp_recipes()  # Remplacez par le chemin de votre fichier

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



