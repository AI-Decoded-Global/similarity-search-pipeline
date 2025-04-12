"""
Functions for generating embeddings from volunteer descriptions.
"""
from sentence_transformers import SentenceTransformer
from typing import List

# Cache loaded models
_model_cache = {}

def get_embedding_model(model_name: str = "all-mpnet-base-v2") -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def generate_embeddings(texts: List[str], model_name: str = "all-mpnet-base-v2") -> List[List[float]]:
    model = get_embedding_model(model_name)
    return model.encode(texts, convert_to_numpy=True)   
