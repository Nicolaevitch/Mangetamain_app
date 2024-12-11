# Mangetamain_app
application mangetamain
Bienvenue sur l'application Mangetamain ! 

## Utilisation
Accédez à l'application directement via ce lien :

👉 **[Mangetamain.app](https://mangetamainapp-main.streamlit.app/)**
## Installation
Si vous souhaitez exécuter ce projet localement :

1. Clonez le dépôt GitHub :
   ```bash
   git clone https://github.com/Nicolaevitch/Mangetamain_app.git

2. Executer poetry install sur votre terminal local lié à votre github.

## Exécution locale
Pour lancer l'application localement :

1. Exécutez la commande suivante :
   ```bash
   streamlit run src/app.py
   ```
2. Ouvrez le lien local généré dans votre navigateur.

## Déploiement
L'application est déployée sur Streamlit Cloud. Si vous souhaitez effectuer des modifications :

1. Poussez vos changements sur le dépôt GitHub.
2. Les mises à jour seront automatiquement déployées.

### Fonctionnalités principales :
- Explorez des recettes interactives.
- Filtrez et recherchez des recettes par ingrédients et catégories.
- Visualisez des données et analyses avancées :             
Le TSNE est un algorithme permettant de réduire la dimension d’une matrice tout en préservant les informations importantes contenues à l’intérieur. Il s'agit d'une technique non linéaire bien adaptée à l'intégration de données à haute dimension pour la visualisation dans un espace à basse dimension. Elle modélise chaque objet par un point de manière à ce que les objets similaires soient modélisés par des points proches et que les objets dissemblables soient modélisés par des points éloignés avec une probabilité élevée.

- Recherche de recettes proches :
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

