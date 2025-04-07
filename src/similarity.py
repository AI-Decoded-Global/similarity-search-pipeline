"""
Functions for finding similar volunteer descriptions.
"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def adjust_for_negation(query: str, candidates: List[str], scores: np.ndarray) -> np.ndarray:
    """
    Lightweight adjustment to emphasize negation in results.
    Works WITH the model's understanding rather than against it.
    """
    # Detect negative phrases (3 words after negation triggers)
    neg_triggers = {'not', 'no', 'except', 'without', 'hate'}
    words = query.lower().split()
    negative_terms = ['hate','dispise']
    
    for i, word in enumerate(words):
        if word in neg_triggers and i+1 < len(words):
            negative_terms.extend(words[i+1:i+4])
    
    # Only adjust if negation was clearly detected
    if not negative_terms:
        return scores
    
    # Light penalty for candidates containing negated terms
    adjusted_scores = scores.copy()
    for i, text in enumerate(candidates):
        if any(term in text for term in negative_terms):
            adjusted_scores[i] *= 0.7  # Moderate 30% penalty
    
    return adjusted_scores

def find_similar_descriptions(
    query: str,
    df: pd.DataFrame,
    model_name: str = 'all-MiniLM-L6-v2',
    top_k: int = 3
) -> pd.DataFrame:
    """
    Finds the top K most similar descriptions to a given query using cosine similarity.

    Returns a DataFrame with:
    - Volunteer ID
    - Original Description
    - Cleaned Description
    - Similarity Score
    """
    # Encode the query
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0]

    # Convert stored list embeddings into a numpy array
    embeddings = np.array(df['embeddings'].tolist())

    # Calculate cosine similarity
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Get top K indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Build results DataFrame
    results = pd.DataFrame({
        'VolunteerID': df.iloc[top_indices]['Volunteer_ID'].values,
        'Original Description': df.iloc[top_indices]['Description'].values,
        'Cleaned Description': df.iloc[top_indices]['description_clean'].values,
        'Similarity Score': similarities[top_indices]
    })

    return results