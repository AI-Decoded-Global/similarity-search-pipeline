import sys
import os
import chromadb
from chromadb.config import Settings
from data_loader import load_and_clean_data
from embedding import EmbeddingGenerator


file_path = "data/volunteer-descriptions.csv"
ids, original_texts, cleaned_texts = load_and_clean_data(file_path)

model_name = "all-MiniLM-L6-v2"  
embedder = EmbeddingGenerator(model_name)

# Generate embeddings
embeddings = embedder.generate_embeddings(cleaned_texts)

# Setup ChromaDB with persistence
chroma_client = chromadb.Client(Settings(
    persist_directory="chroma_db",  # Where ChromaDB will save data
    anonymized_telemetry=False
))

# Create or get the collection
collection = chroma_client.get_or_create_collection(name="volunteers")

# Convert volunteer IDs to strings (Chroma requires string IDs)
string_ids = [str(vol_id) for vol_id in ids]

# Create metadata records
metadata = [{"volunteer_id": vol_id} for vol_id in ids]

# Add data to the ChromaDB collection
collection.add(
    ids=string_ids,
    documents=original_texts,
    embeddings=embeddings,
    metadatas=metadata
)

# # Persist the data
# chroma_client.persist()




