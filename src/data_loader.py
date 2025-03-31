import pandas as pd
import re

def clean_text(text: str) -> str:
    """
    Takes in input text, lowers the strings and remove any whitespaces
    """
    
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.strip()


def load_and_clean_data(file_path: str, text_column: str = "Description", id_column: str = "Volunteer_ID"):
    """
    Loads a CSV file, cleans it, converts columns to lists. 

    Returns:
        Lists: Lists of ids, original text and cleaned text 
    """

    df = pd.read_csv(file_path)
    df["cleaned_text"] = df[text_column].apply(clean_text)
    df = df.dropna()
    df = df.drop_duplicates()

    texts = df[text_column].tolist()
    cleaned_texts = df["cleaned_text"].tolist()
    ids = df[id_column].tolist()

    return ids, texts, cleaned_texts
