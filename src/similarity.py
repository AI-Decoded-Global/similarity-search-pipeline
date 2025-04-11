"""
Functions for finding similar volunteer descriptions.
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.nn.functional import cosine_similarity

from src.embedding import generate_embeddings
from src.data_loader import clean_text

# Optional imports for FAISS and ChromaDB
try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    chromadb = None

class SimilarityEngine:
    def __init__(self, df, embeddings, backend="in_memory"):
        self.df = df
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.backend = backend

        if backend == "faiss":
            # pass
            if faiss is None:
                raise ImportError("FAISS is not installed.")
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
            print('fiass')

        elif backend == "chromadb":
            # pass
            if chromadb is None:
                raise ImportError("ChromaDB is not installed.")
            self.client = chromadb.Client(Settings(anonymized_telemetry=False))
            self.collection = self.client.create_collection(name="volunteer_embeddings")
            self.collection.add(
                embeddings=self.embeddings.tolist(),
                ids=[str(i) for i in self.df["Volunteer_ID"].tolist()],
                documents=self.df["Description"].tolist()
            )
        else:
            # In-memory setup — no index needed
            pass


    def query(self, query_text: str, top_k: int = 3, model_name: str = "all-mpnet-base-v2") -> list:

        query_text = clean_text(query_text)
        query_embedding = generate_embeddings([query_text], model_name)
        query_tensor = torch.tensor(query_embedding, dtype=torch.float32)

        if self.backend == "faiss":
            D, I = self.index.search(np.array([query_embedding[0]]), top_k)
            results = [
                {
                    "volunteer_id": int(self.df.iloc[idx]["Volunteer_ID"]),
                    "description": self.df.iloc[idx]["Description"],
                    "score": float(1+(1 - D[0][i]))  # convert L2 to similarity
                }
                for i, idx in enumerate(I[0])
            ]

        elif self.backend == "chromadb":
            results_raw = self.collection.query(
                query_embedding[0],
                n_results=top_k
            )
            results = []
            for i in range(top_k):
                results.append({
                    "volunteer_id": results_raw["ids"][0][i],
                    "description": results_raw["documents"][0][i],
                    "score": float(1+(1-results_raw["distances"][0][i]))  # similarity score
                })

        else:
            similarities = cosine_similarity(query_tensor, self.embeddings)
            top_k_indices = torch.topk(similarities, top_k).indices.tolist()

            results = []
            for idx in top_k_indices:
                results.append({
                    "volunteer_id": int(self.df.iloc[idx].get("Volunteer_ID", idx)),
                    "description": self.df.iloc[idx]["Description"],
                    "similarity": float(similarities[idx])
                })
        return results





