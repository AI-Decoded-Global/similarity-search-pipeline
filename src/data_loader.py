"""
Functions for loading and preprocessing volunteer data.
"""

# ######################################################################################
#                                       Imports
# ######################################################################################

# %%
# import math
# import numpy as np
# import datetime as dt
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# # %%
# #  Let's read in the Volunteer Description data
# df = pd.read_csv('../data/volunteer-descriptions.csv', sep=',')

# # %%
# # Let's see what we're working with
# df.head(5)

# ######################################################################################
#                                       Input variables
# ######################################################################################

filepath = '../data/volunteer-descriptions.csv'

# Let's specify some switches that we can use to clean the data depending on user input. This can also be added as a config file later.
apply_lower_case = True
apply_remove_punc = True # remove all punctution
apply_remove_whitespace = True # this will remove multiple spces, /n, /t, and strip spaces on both ends of the String
apply_remove_stopwords = False
apply_stemming = False
# apply_lemmatization = False  # Let's just use stemming for this MVP
apply_abbreviation_map = True
apply_term_standardization = True
        
custom_abbreviation_map = {
    # Days of week
    'm': 'monday',
    'mo': 'monday',
    'mon': 'monday',
    't': 'tuesday',
    'tu': 'tuesday',
    'tue': 'tuesday',
    'w': 'wednesday',
    'wed': 'wednesday',
    'th': 'thursday',
    'thu': 'thursday',
    'f': 'friday',
    'fr': 'friday',
    'fri': 'friday',
    'sa': 'saturday',
    'su': 'sunday',
    
    # Time indicators
    'pm': 'evening',
    'hrs': 'hours',
    'yrs': 'years',
    
    # Common terms
    'orgs': 'organizations',
    'nonprofits': 'nonprofit organizations',
    'fb': 'facebook',
    'insta': 'instagram',
    'dm': 'direct message',
    'u': 'you',
    'ur': 'your',
    'plz': 'please',
    'prob': 'probably',
    'esp': 'especially',
    
    # Technical terms
    'html': 'HTML',
    'css': 'CSS',
    'js': 'javascript',
    'c++': 'C++',
    'php': 'PHP',
    'seo': 'search engine optimization',
    'ppc': 'pay-per-click',
    'ux': 'user experience',
    'pmp': 'Project Management Professional',
    
    # Emergency/medical
    'cpr': 'cardiopulmonary resuscitation',
    'aed': 'automated external defibrillator',
    'ems': 'emergency medical services',
    
    # Education
    'esl': 'english as a second language',
    'stem': 'science technology engineering math',
    # etc... can add more to this list
}

custom_term_standardization = {
    'old people':'senior',
    'elderly':'senior',
    'older folks':'senior',
    'kid':'child',
    'childs':'child',
    'youth':'child'   
    # etc... 
}


# %%
# ######################################################################################
#                                       Functions - Load & Clean
# ######################################################################################

def load_data(file_path: str, sep: str = ',', encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Loads the CSV data into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.
    - sep (str): The delimiter used in the CSV file. Default is comma (',').
    - encoding (str): The encoding used to read the file. Default is 'utf-8'.
    
    Returns:
    - DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding)
        print(f"Data loaded successfully from {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    except pd.errors.ParserError:
        print(f"Error: There was an issue with the CSV format in {file_path}.")
        return pd.DataFrame()  # Return an empty DataFrame in case of error
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


def f_handle_contractions(text: str) -> str:
    # Let's expand some common contractions before other cleaning
    contraction_patterns = [
        (r"(\w+)'ve", r'\1 have'),
        (r"(\w+)'ll", r'\1 will'),
        (r"(\w+)'re", r'\1 are'),
        (r"(\w+)'d", r'\1 would'),
        (r"(\w+)'s", r'\1 is'),
        (r"(\w+)'m", r'\1 am'),
        (r"can't", 'cannot'),
        (r"won't", 'will not')
    ]
    
    for pattern, replacement in contraction_patterns:
        text = re.sub(pattern, replacement, text)
    return text

def f_clean_text(text:str) -> str:
    if pd.isna(text):
        return ""  
    
    if apply_lower_case:
        text = str(text).lower()  # convert text to lowercase
        
    if apply_remove_punc:
        text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuation
        
    if apply_remove_whitespace:
        text = re.sub(r'\s+', ' ', text)  # remove multiple whitespace within the text
        text = text.strip() # remove spaces on both ends of string
    
    return text

def f_apply_abbr_map(text: str) -> str:
    if apply_abbreviation_map:
        for abbrev, full in custom_abbreviation_map.items():
            # Use word boundaries to avoid partial matches
            text = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, text)
    return text

def f_apply_term_stand(text:str) -> str:
    if apply_term_standardization:
        for term, standardized in custom_term_standardization.items():
            text = text.replace(term, standardized)
    return text

def f_apply_remove_stopwords(text:str) -> str:
    if apply_remove_stopwords:
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        return ' '.join(filtered_words)
    else:
        return text

def f_apply_stemming(text:str) -> str:
    if apply_stemming:
        stemmer = PorterStemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    else:
        return text

def full_clean_pipeline(text:str) -> str:
    text = f_handle_contractions(text)
    text = f_clean_text(text)
    text = f_apply_abbr_map(text)
    text = f_apply_term_stand(text)
    text = f_apply_remove_stopwords(text)
    text = f_apply_stemming(text)
    
    return text
