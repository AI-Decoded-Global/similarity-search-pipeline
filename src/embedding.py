"""
Functions for generating embeddings from volunteer descriptions.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

def embed_texts(texts: List[str], model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Embeds a list of cleaned text descriptions using a pre-trained SentenceTransformer model.

    Parameters:
    - texts (List[str]): List of cleaned text strings.
    - model_name (str): The pre-trained model to use for embedding.

    Returns:
    - np.ndarray: Matrix of embeddings (shape: [len(texts), embedding_dim])
    """
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} texts with model '{model_name}'...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def add_embeddings_to_dataframe(df: pd.DataFrame, text_column: str = 'description_clean_full', model_name: str = 'all-MiniLM-L6-v2') -> pd.DataFrame:
    """
    Adds embeddings as a new column in the DataFrame for a specified text column.

    Parameters:
    - df (pd.DataFrame): The DataFrame with the cleaned text.
    - text_column (str): The name of the column containing cleaned text.
    - model_name (str): The pre-trained model to use for embedding.
    
    Returns:
    - pd.DataFrame: The DataFrame with an added 'embeddings' column.
    """
    if text_column in df.columns:
        # Embed the descriptions and add them to a new 'embeddings' column
        embeddings = embed_texts(df[text_column].tolist(), model_name)
        df['embeddings'] = list(embeddings)  # Convert numpy array to list for storage in DataFrame
        print(f"Embeddings added to DataFrame with shape: {embeddings.shape}")
    else:
        print(f"Error: Column '{text_column}' not found in the DataFrame.")
    return df