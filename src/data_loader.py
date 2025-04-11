"""
Functions for loading and preprocessing volunteer data.
"""
import pandas as pd
import re
import nltk
import string
from langdetect import detect
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download('wordnet')
# lemmatizer = WordNetLemmatizer()

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def translate_to_english(text, source_lang):
    if source_lang == "en" or source_lang == "unknown":
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception as e:
        print(f"Translation failed: {e}")
        return text
    
def remove_stopwords(text):
    try:
        nltk.data.find('/Users/m1pro/nltk_data/corpora/stopwords.zip')
    except LookupError:
        print ('downloading nltk resources... One time download')
        nltk.download('stopwords')
    
    # stop_words = set(stopwords.words('english'))
    stops = set(stopwords.words('english'))
    tokens = text.split()
    filtered = [word for word in tokens if word not in stops]
    return ' '.join(filtered)

def lemmatize(text):
    try:
        nltk.data.find('/Users/m1pro/nltk_data/corpora/wordnet.zip')
    except LookupError:
        print ('downloading nltk resources... One time download')
        nltk.download('wordnet')

    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove punctuation/special chars
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()    
    # detect language and translate to english 
    lang = detect_language(text)
    text = translate_to_english(text, lang)
    # remove stopwords and lemmatize  
    text = remove_stopwords(text)
    text = lemmatize(text)
    return text

def load_and_clean_data(df) -> pd.DataFrame:
    # df = pd.read_csv(path)
    if 'Description' not in df.columns or 'Volunteer_ID' not in df.columns:
        raise ValueError("CSV must have 'Volunteer_ID' and 'Description' columns.")
    df['cleaned_description'] = df['Description'].astype(str).apply(clean_text)
    return df