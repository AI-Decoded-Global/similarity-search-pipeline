import json
from embedding import embedder
from similarity import SimilaritySearch
from data_loader import clean_text
import pickle

with open("similarity-search-pipeline/data/embeddings.pkl", "rb") as f:
    ids, original_texts, cleaned_texts, embeddings = pickle.load(f)


def search_similarity(query: str, model_name):
    """
    Takes in a query and embedding model,
    uses cosine similarity to compare the distance between embedded documents and 
    embeddings of a query string to return results docs similar to the query of the user. 

    Args:
        query(str): input query of a user
        model: hugging face/AI model for similarity search

    Returns:
        Json: json object with description, volunteerID and similarity score
    """
    
    # Create searcher and run similarity
    searcher = SimilaritySearch(ids, original_texts, embeddings)
    results = searcher.find_similar(query, embedder.model, clean_text)

    return json.dumps({"model": model_name, "Top_Matches": results}, indent=4)

if __name__ == "__main__":
    query = "Looking for volunteers skilled in graphic design to help with non-profit branding."
    print(search_similarity(query, "all-MiniLM-L6-v2"))
