import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
from data_loader import clean_text
from create_chroma_db import chroma_client


collection = chroma_client.get_collection("volunteers")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_chroma(query: str, top_k: int = 3):
    query_clean = clean_text(query)
    query_embedding = model.encode([query_clean], convert_to_numpy=True)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    results_json = []

    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        results_json.append({
            "Volunteer_ID": meta['volunteer_id'],
            "Description": doc,
            "Similarity_score": round(dist, 2)
        })
   
    print((json.dumps({"Top_matches": results_json}, indent=4)))

# Example
search_chroma("Looking for volunteers skilled in graphic design to help with non-profit branding.")

