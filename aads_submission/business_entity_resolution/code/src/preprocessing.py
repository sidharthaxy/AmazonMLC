import pandas as pd
import re
import string

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads TSV data explicitly with string dtypes.
    """
    return pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

def clean_text(text: str) -> str:
    """
    Normalizes uppercase/lowercase, removes extraneous punctuation, and cleans whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace punctuation with space to separate words properly
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_abbreviations(text: str) -> str:
    """
    Expands legal/business and address abbreviations.
    """
    if not text:
        return ""
    
    # Pad with spaces to match whole words only
    text = " " + text + " "
    
    replacements = {
        r'\bpvt\b': 'private',
        r'\bltd\b': 'limited',
        r'\bcorp\b': 'corporation',
        r'\binc\b': 'incorporated',
        r'\b&\b': 'and',
        r'\brd\b': 'road',
        r'\bst\b': 'street',
        r'\bave\b': 'avenue',
        r'\bllc\b': 'limited liability company',
        r'\bco\b': 'company'
    }
    
    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)
        
    return text.strip()

def extract_numerical_tokens(text: str) -> str:
    """
    Extracts numerical tokens (e.g. zip codes, building numbers) from text.
    Returns them as a space-separated string.
    """
    if not text:
        return ""
    # Find all consecutive digits
    numbers = re.findall(r'\d+', text)
    return " ".join(numbers)

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all preprocessing steps to the dataframe.
    """
    df = df.copy()
    
    if 'business_name' in df.columns:
        df['business_name_clean'] = df['business_name'].apply(clean_text)
        df['business_name_norm'] = df['business_name_clean'].apply(normalize_abbreviations)
        df['name_numbers'] = df['business_name_clean'].apply(extract_numerical_tokens)
        
    if 'business_address' in df.columns:
        df['business_address_clean'] = df['business_address'].apply(clean_text)
        df['business_address_norm'] = df['business_address_clean'].apply(normalize_abbreviations)
        df['address_numbers'] = df['business_address_clean'].apply(extract_numerical_tokens)
        
    if 'country' in df.columns:
        df['country_clean'] = df['country'].apply(clean_text)
        
    return df
