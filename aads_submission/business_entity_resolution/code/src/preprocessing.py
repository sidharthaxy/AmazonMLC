import pandas as pd
import re
import string

ABBR_MAP = {
    'pvt': 'private',
    'ltd': 'limited',
    'corp': 'corporation',
    'inc': 'incorporated',
    'rd': 'road',
    'st': 'street',
    'ave': 'avenue',
    'llc': 'limited liability company',
    'co': 'company'
}
ABBR_PATTERN = re.compile(r'\b(' + '|'.join(ABBR_MAP.keys()) + r')\b')
NUM_PATTERN = re.compile(r'\d+')
WHITESPACE_PATTERN = re.compile(r'\s+')
PUNCT_TRANS = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads TSV data explicitly with string dtypes.
    """
    return pd.read_csv(filepath, sep="\t", dtype=str, keep_default_na=False)

def clean_text(text: str) -> str:
    """
    Normalizes uppercase/lowercase, removes punctuation, and cleans whitespace.
    """
    if not isinstance(text, str) or not text:
        return ""
    text = text.lower()
    text = text.translate(PUNCT_TRANS)
    text = WHITESPACE_PATTERN.sub(' ', text).strip()
    return text

def normalize_abbreviations(text: str) -> str:
    """
    Expands legal/business and address abbreviations using compiled regex.
    """
    if not text:
        return ""
    return ABBR_PATTERN.sub(lambda m: ABBR_MAP[m.group(0)], text)

def extract_numerical_tokens(text: str) -> str:
    """
    Extracts numerical tokens (e.g. zip codes, building numbers) from text.
    Returns them as a space-separated string.
    """
    if not text:
        return ""
    numbers = NUM_PATTERN.findall(text)
    return " ".join(numbers)

def preprocess_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Applies text cleaning and normalization to dataframe columns.
    """
    if not inplace:
        df = df.copy()
        
    if 'business_name' in df.columns:
        df['business_name_clean'] = [clean_text(t) for t in df['business_name'].values]
        df['business_name_norm'] = [normalize_abbreviations(t) for t in df['business_name_clean'].values]
        
    if 'business_address' in df.columns:
        df['business_address_clean'] = [clean_text(t) for t in df['business_address'].values]
        df['business_address_norm'] = [normalize_abbreviations(t) for t in df['business_address_clean'].values]
        df['address_numbers'] = [extract_numerical_tokens(t) for t in df['business_address_clean'].values]
        
    if 'country' in df.columns:
        df['country_clean'] = [clean_text(t) for t in df['country'].values]
        
    return df
