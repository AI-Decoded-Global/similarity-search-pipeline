# similarity.py

from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict
import numpy as np

class SimilaritySearch:
    def __init__(self, ids: List[int], original_texts: List[str], embeddings: np.ndarray):
        """
        Initializes the SimilaritySearch class with volunteer IDs, original descriptions, 
        and their corresponding embeddings.

        Args:
            ids (List[int]): List of volunteer IDs.
            original_texts (List[str]): List of original volunteer descriptions.
            embeddings (np.ndarray): Embeddings generated from cleaned descriptions.
        """
        self.ids = ids
        self.texts = original_texts
        self.embeddings = embeddings

    def find_similar(self, query: str, model, clean_func, top_k: int = 3) -> List[Dict]:
        """
        Computes cosine similarity between the query and the embedded documents,
        and returns the top_k most similar results.

        Args:
            query (str): Input user query.
            model: SentenceTransformer model used for embedding the query.
            clean_func: Function to clean the query text (same as used in preprocessing).
            top_k (int): Number of top matches to return.

        Returns:
            List[Dict]: A list of dictionaries with volunteer ID, description,
                        and similarity score for top_k similar results.
        """

        # Clean the input query using the same preprocessing function
        query_clean = clean_func(query)

        # Generate embedding for the cleaned query
        query_embedding = model.encode([query_clean], convert_to_numpy=True)

        # Compute cosine similarity between the query and all stored embeddings
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get the indices of top_k most similar entries in descending order
        top_indices = similarities.argsort()[::-1][:top_k]

        # Build and return the list of results
        results = []
        for idx in top_indices:
            results.append({
                "Volunteer_ID": self.ids[idx],
                "Description": self.texts[idx],
                "Similarity_Score": round(float(similarities[idx]), 2)
            })

        return results
