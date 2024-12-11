import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from recipe_app import RecipeApp, IngredientDataError

class TestLoadMainData(unittest.TestCase):
    def test_load_main_data_success(self):
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame({'id': [1, 2], 'name': ['Recipe 1', 'Recipe 2']})
            app = RecipeApp()
            self.assertFalse(app.recipes_clean.empty)
            self.assertIn('id', app.recipes_clean.columns)

    def test_load_main_data_file_not_found(self):
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            with self.assertRaises(IngredientDataError):
                RecipeApp()

class TestGetIngredientsData(unittest.TestCase):
    def test_get_ingredients_data_success(self):
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [
                pd.DataFrame({'id': [1], 'ingredients': ['["sugar", "flour"]']}),
                pd.DataFrame({'id': [2], 'ingredients': ['["butter", "milk"]']})
            ]
            app = RecipeApp()
            result = app.get_ingredients_data()
            self.assertEqual(len(result), 2)
            self.assertIn('ingredients', result.columns)

    def test_get_ingredients_data_file_not_found(self):
        with patch('pandas.read_csv', side_effect=FileNotFoundError):
            app = RecipeApp()
            with self.assertRaises(IngredientDataError):
                app.get_ingredients_data()

class TestFilterRecipes(unittest.TestCase):
    def test_filter_recipes_success(self):
        with patch.object(RecipeApp, 'get_ingredients_data') as mock_ingredients_data:
            mock_ingredients_data.return_value = pd.DataFrame({
                'id': [1, 2],
                'ingredients': ['["sugar", "flour"]', '["butter", "milk"]']
            })
            app = RecipeApp()
            app.recipes_clean = pd.DataFrame({'id': [1, 2], 'name': ['Recipe 1', 'Recipe 2']})
            result = app.filter_recipes(['sugar'])
            self.assertEqual(len(result), 1)
            self.assertEqual(result.iloc[0]['id'], 1)

    def test_filter_recipes_empty_selection(self):
        app = RecipeApp()
        result = app.filter_recipes([])
        self.assertTrue(result.empty)

    def test_filter_recipes_invalid_data(self):
        with patch.object(RecipeApp, 'get_ingredients_data') as mock_ingredients_data:
            mock_ingredients_data.return_value = pd.DataFrame({
                'id': [1],
                'ingredients': ['invalid_data']
            })
            app = RecipeApp()
            result = app.filter_recipes(['sugar'])
            self.assertTrue(result.empty)

class TestDisplayMacroIngredientsMenu(unittest.TestCase):
    @patch('streamlit.multiselect')
    def test_display_macro_ingredients_menu(self, mock_multiselect):
        mock_multiselect.return_value = ['sugar', 'flour']
        app = RecipeApp()
        result = app.display_macro_ingredients_menu()
        self.assertEqual(result, ['sugar', 'flour'])

class TestDisplayFilteredRecipes(unittest.TestCase):
    @patch('streamlit.dataframe')
    def test_display_filtered_recipes_with_data(self, mock_dataframe):
        with patch.object(RecipeApp, 'filter_recipes') as mock_filter_recipes:
            mock_filter_recipes.return_value = pd.DataFrame({'id': [1], 'name': ['Recipe 1']})
            app = RecipeApp()
            app.display_filtered_recipes(['sugar'])
            mock_dataframe.assert_called_once()

    @patch('streamlit.title')
    def test_display_filtered_recipes_no_data(self, mock_title):
        with patch.object(RecipeApp, 'filter_recipes') as mock_filter_recipes:
            mock_filter_recipes.return_value = pd.DataFrame()
            app = RecipeApp()
            app.display_filtered_recipes(['sugar'])
            mock_title.assert_called_once_with("On est pas des cakes !")
