# vectorizers.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

class Vectorizer:
    @staticmethod
    def tfidf_vectorize(column_data):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(column_data)
        return tfidf_matrix, vectorizer  # Retourner la matrice sparse et le vectorizer

    @staticmethod
    def bow_vectorize(column_data):
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(column_data)
        return bow_matrix, vectorizer  # Retourner la matrice sparse et le vectorizer
