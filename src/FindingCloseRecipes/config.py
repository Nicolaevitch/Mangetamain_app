# config.py
NUMERIC_FEATURES = [
    'log_minutes',
    'calories',
    'total fat (PDV%)',
    'sugar (PDV%)',
    'sodium (PDV%)',
    'protein (PDV%)',
    'saturated fat (PDV%)',
    'carbohydrates (PDV%)'
]

DEFAULT_WEIGHTS = {
    'log_minutes': 0.5,
    'calories': 0.2,
    'total fat (PDV%)': 0.05,
    'sugar (PDV%)': 0.05,
    'sodium (PDV%)': 0.05,
    'protein (PDV%)': 0.05,
    'saturated fat (PDV%)': 0.05,
    'carbohydrates (PDV%)': 0.05
}

COMBINED_WEIGHTS = {
    "alpha": 0.05,  # 'name'
    "beta": 0.3,    # 'tags'
    "gamma": 0.3,   # 'steps'
    "delta": 0.3,   # 'ingredients'
    "epsilon": 0.05  # numeric features
}

TOP_N = 100  # Nombre de recettes similaires

