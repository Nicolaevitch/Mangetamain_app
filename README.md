# Mangetamain_app
application mangetamain
Bienvenue sur l'application Mangetamain ! 
Quatre pages de recettes s'offrent à toi :

-Page d'accueil 
Elle te permet de suivre ton activité grâce à ton id.

-Idée recette ! 
En renseignant des ingrédients, tu trouveras des idées de recettes contenant les 
ingrédients choisis. 

-Représentation de recette

-Recherche de recettes proches :
A l'adresse de la personne en charge de l'application:

Lorsque de nouvelles recettes sont arrivées dans le fichier Raw_recipes.csv , vous pouvez lancer le fichier run_preprocessing.py
Ce preprocessing est nécessaire pour le calcul des distances entre recettes et suit ces étapes:

- Suppression des recettes avec des valeurs aberrantes (temps de préparation excessif, calories élevées).
- Gestion des valeurs manquantes et des colonnes inutiles.
- Remplacement des ingrédients par des catégories standardisées grâce à un fichier de mapping.
- Extraction et nettoyage des caractéristiques nutritionnelles.
- Transformation logarithmique des temps de préparation.
- Suppression des colonnes inutiles pour l'analyse.
- Nettoie et transforme les colonnes name, steps, ingredients et tags.
- Applique des techniques comme le stemming et la suppression des stop words.
- Normalise les colonnes numériques à l'aide de StandardScaler, garantissant une mise à l'échelle cohérente pour l'analyse.
- Divise le dataset final en plusieurs fichiers CSV pour une gestion via Streamlit Cloud
- Division par colonnes textuelles ou numériques.
- Découpage en plusieurs parties pour les grands fichiers.
