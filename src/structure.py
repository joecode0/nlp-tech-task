import pandas as pd
import numpy as np
from scipy import sparse
import os

def save_structured_data(features: dict,
                         processed_path: str,
                         tfidf_matrix_path: str,
                         tfidf_terms_path: str,
                         embeddings_path: str):
    """
    Save outputs from the feature_engineering pipeline to disk.

    Args:
        features (dict): Dictionary from feature_engineering().
        processed_path (str): Path to save enriched DataFrame (.csv).
        tfidf_path (str): Path to save TF-IDF matrix (.npz) — terms saved as _terms.csv.
        embeddings_path (str): Path to save sentence embeddings (.npy).
    """
    # Ensure output directories exist
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(tfidf_matrix_path), exist_ok=True)
    os.makedirs(os.path.dirname(tfidf_terms_path), exist_ok=True)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)

    # Save enriched DataFrame
    features["df"].to_csv(processed_path, index=False)

    # Save TF-IDF matrix and terms
    sparse.save_npz(tfidf_matrix_path, features["tfidf_matrix"])
    pd.Series(features["tfidf_terms"]).to_csv(tfidf_terms_path, index=False)

    # Save embeddings
    np.save(embeddings_path, features["embeddings"])

    print("Saved:")
    print(f"- DataFrame: {processed_path}")
    print(f"- TF-IDF Matrix: {tfidf_matrix_path}")
    print(f"- TF-IDF Terms: {tfidf_terms_path}")
    print(f"- Embeddings: {embeddings_path}")