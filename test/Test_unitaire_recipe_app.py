import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from recipe_app import RecipeApp, IngredientDataError


class TestRecipeApp(unittest.TestCase):
    """Tests unitaires pour les fonctionnalités de la classe RecipeApp."""

    def setUp(self):
        """Initialisation avec des mocks pour les données."""
        with patch('recipe_app.RecipeApp.load_main_data') as mock_load_main_data:
            mock_load_main_data.return_value = pd.DataFrame({
                'id': [1, 2],
                'name': ['Recipe 1', 'Recipe 2']
            })
            self.app = RecipeApp()

    @patch('pandas.read_csv')
    def test_load_main_data_success(self, mock_read_csv):
        """Test que les données principales sont chargées correctement."""
        mock_read_csv.return_value = pd.DataFrame({'id': [1, 2], 'name': ['Recipe 1', 'Recipe 2']})
        result = self.app.load_main_data()
        self.assertFalse(result.empty)
        self.assertIn('id', result.columns)

    @patch('pandas.read_csv', side_effect=FileNotFoundError)
    def test_load_main_data_failure(self, mock_read_csv):
        """Test qu'une exception est levée si le fichier principal est introuvable."""
        with self.assertRaises(IngredientDataError):
            self.app.load_main_data()

    @patch('pandas.read_csv')
    def test_get_ingredients_data_success(self, mock_read_csv):
        """Test que les données d'ingrédients sont chargées et combinées correctement."""
        mock_read_csv.side_effect = [
            pd.DataFrame({'id': [1], 'ingredients': ['["sugar", "flour"]']}),
            pd.DataFrame({'id': [2], 'ingredients': ['["butter", "milk"]']}),
        ]
        result = self.app.get_ingredients_data()
        self.assertEqual(len(result), 2)
        self.assertIn('ingredients', result.columns)

    @patch('pandas.read_csv', side_effect=FileNotFoundError)
    def test_get_ingredients_data_failure(self, mock_read_csv):
        """Test qu'une exception est levée si les fichiers d'ingrédients sont introuvables."""
        with self.assertRaises(IngredientDataError):
            self.app.get_ingredients_data()

    @patch.object(RecipeApp, 'get_ingredients_data')
    def test_filter_recipes_success(self, mock_ingredients_data):
        """Test que les recettes sont correctement filtrées selon les ingrédients."""
        mock_ingredients_data.return_value = pd.DataFrame({
            'id': [1, 2],
            'ingredients': ['["sugar", "flour"]', '["butter", "milk"]']
        })
        self.app.recipes_clean = pd.DataFrame({'id': [1, 2], 'name': ['Recipe 1', 'Recipe 2']})
        result = self.app.filter_recipes(['sugar'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]['id'], 1)

    @patch.object(RecipeApp, 'get_ingredients_data')
    def test_filter_recipes_no_match(self, mock_ingredients_data):
        """Test que le filtrage retourne un DataFrame vide si aucun ingrédient ne correspond."""
        mock_ingredients_data.return_value = pd.DataFrame({
            'id': [1, 2],
            'ingredients': ['["sugar", "flour"]', '["butter", "milk"]']
        })
        self.app.recipes_clean = pd.DataFrame({'id': [1, 2], 'name': ['Recipe 1', 'Recipe 2']})
        result = self.app.filter_recipes(['chocolate'])
        self.assertTrue(result.empty)

    @patch('streamlit.multiselect')
    def test_display_macro_ingredients_menu(self, mock_multiselect):
        """Test que la méthode retourne les ingrédients sélectionnés par l'utilisateur."""
        mock_multiselect.return_value = ['sugar', 'flour']
        result = self.app.display_macro_ingredients_menu()
        self.assertEqual(result, ['sugar', 'flour'])


if __name__ == '__main__':
    unittest.main()
