import pandas as pd
import re
import string
import unicodedata

# ─────────────────────────────────────────────────────────────────────────────
# Legal suffix removal patterns (international, sorted longest-first)
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_SUFFIXES_RAW = [
    # Multi-word (must come first in alternation)
    'private limited', 'pvt limited', 'pvt ltd',
    'limited liability company', 'limited liability partnership',
    'societe anonyme', 'societe a responsabilite limitee',
    'societe par actions simplifiee',
    'societe civile immobiliere',
    'gesellschaft mit beschrankter haftung',
    # Single-word English
    'incorporated', 'inc', 'corporation', 'corp', 'company', 'co',
    'limited', 'ltd', 'llc', 'llp', 'lp', 'plc',
    'holdings', 'holding',
    # French
    'sa', 'sas', 'sarl', 'sasu', 'eurl', 'sci', 'snc', 'sem',
    # German
    'gmbh', 'ag', 'kg', 'ohg', 'ug', 'ev',
    # Indian
    'pvt', 'private', 'nidhi', 'opc',
    # Other
    'pty', 'bv', 'nv', 'ab', 'as', 'oy', 'srl', 'spa',
    'pte',
]
# Sort longest-first so multi-word suffixes match before their fragments
LEGAL_SUFFIXES_RAW.sort(key=len, reverse=True)

LEGAL_SUFFIX_PATTERN = re.compile(
    r'(?:^|\s)(?:' + '|'.join(re.escape(s) for s in LEGAL_SUFFIXES_RAW) + r')(?:\s*\.?\s*,?\s*$|\s)',
    re.IGNORECASE
)

# Abbreviation expansion (kept for backward compatibility)
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

# Ampersand normalization
AMPERSAND_PATTERN = re.compile(r'\s*&\s*')


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


def strip_legal_suffixes(text: str) -> str:
    """
    Removes legal/business suffixes from a cleaned business name.
    Produces the 'core' business identity for high-precision matching.
    E.g., "abc solutions pvt ltd" → "abc"
    """
    if not text:
        return ""
    # Iteratively strip suffixes (may need multiple passes for compound suffixes)
    prev = ""
    result = text
    for _ in range(3):
        if result == prev:
            break
        prev = result
        result = LEGAL_SUFFIX_PATTERN.sub(' ', result).strip()
    return WHITESPACE_PATTERN.sub(' ', result).strip()


def normalize_ampersand(text: str) -> str:
    """Normalizes '&' to 'and'."""
    if not text:
        return ""
    return AMPERSAND_PATTERN.sub(' and ', text).strip()


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


def normalize_name_for_matching(raw_name: str) -> str:
    """
    Full normalization pipeline for business name matching:
    1. Lowercase + remove punctuation
    2. Normalize ampersand
    3. Strip legal suffixes
    Returns the 'core' business name for matching.
    """
    cleaned = clean_text(raw_name)
    cleaned = normalize_ampersand(cleaned)
    stripped = strip_legal_suffixes(cleaned)
    return stripped if stripped else cleaned


def preprocess_dataframe(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Applies text cleaning and normalization to dataframe columns.
    """
    if not inplace:
        df = df.copy()

    if 'business_name' in df.columns:
        df['business_name_clean'] = [clean_text(t) for t in df['business_name'].values]
        df['business_name_norm'] = [normalize_abbreviations(t) for t in df['business_name_clean'].values]
        df['business_name_stripped'] = [strip_legal_suffixes(t) for t in df['business_name_clean'].values]

    if 'business_address' in df.columns:
        df['business_address_clean'] = [clean_text(t) for t in df['business_address'].values]
        df['business_address_norm'] = [normalize_abbreviations(t) for t in df['business_address_clean'].values]
        df['address_numbers'] = [extract_numerical_tokens(t) for t in df['business_address_clean'].values]

    if 'country' in df.columns:
        df['country_clean'] = [clean_text(t) for t in df['country'].values]

    return df
