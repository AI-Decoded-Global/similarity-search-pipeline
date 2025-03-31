# embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from data_loader import load_and_clean_data
import pickle

class EmbeddingGenerator:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

# Load and clean data once
file_path = 'similarity-search-pipeline/data/volunteer-descriptions.csv'
ids, original_texts, cleaned_texts = load_and_clean_data(file_path)

model_name = "all-MiniLM-L6-v2"  
embedder = EmbeddingGenerator(model_name)

# Generate embeddings
embeddings = embedder.generate_embeddings(cleaned_texts)

with open("similarity-search-pipeline/data/embeddings.pkl", "wb") as f:
    pickle.dump((ids, original_texts, cleaned_texts, embeddings), f)
