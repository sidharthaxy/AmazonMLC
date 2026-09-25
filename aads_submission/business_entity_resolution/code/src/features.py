import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
from typing import List, Tuple

def compute_jaccard(s1: str, s2: str) -> float:
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def get_ngram_cos_sim(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2:
        return 0.0
    
    def get_ngrams(text):
        text = " " + text + " "
        return [text[i:i+n] for i in range(len(text)-n+1)]
        
    ng1 = get_ngrams(s1)
    ng2 = get_ngrams(s2)
    
    if not ng1 or not ng2:
        return 0.0
        
    set1 = set(ng1)
    set2 = set(ng2)
    
    intersection = len(set1.intersection(set2))
    norm = np.sqrt(len(set1)) * np.sqrt(len(set2))
    
    return intersection / norm if norm > 0 else 0.0

def build_feature_dataset(df_candidates: pd.DataFrame, df_s1: pd.DataFrame, df_s2_s3: pd.DataFrame) -> pd.DataFrame:
    """
    df_candidates is expected to have 'source1_entity_id' and 'candidate_entity_ids'.
    df_candidates can be flattened first so each row is a pair (S1, Candidate).
    """
    # Flatten candidates
    pairs = []
    for _, row in df_candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        for cand in cands:
            if cand.strip():
                pairs.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand.strip()})
                
    df_pairs = pd.DataFrame(pairs)
    if df_pairs.empty:
        return pd.DataFrame()
        
    # Merge with attributes
    df_pairs = df_pairs.merge(df_s1, left_on='source1_entity_id', right_on='entity_id', suffixes=('', '_s1'))
    df_pairs = df_pairs.merge(df_s2_s3, left_on='candidate_entity_id', right_on='entity_id', suffixes=('_1', '_2'))
    
    # Feature calculation lists
    features = []
    
    # Pre-calculate token counts to avoid doing it per row
    df_pairs['name_tokens_1'] = df_pairs['business_name_norm_1'].apply(lambda x: len(x.split()))
    df_pairs['name_tokens_2'] = df_pairs['business_name_norm_2'].apply(lambda x: len(x.split()))
    
    for _, row in df_pairs.iterrows():
        name1 = row['business_name_norm_1']
        name2 = row['business_name_norm_2']
        addr1 = row['business_address_norm_1']
        addr2 = row['business_address_norm_2']
        
        # Name features
        lev_ratio = fuzz.ratio(name1, name2)
        jw_dist = distance.JaroWinkler.normalized_similarity(name1, name2) * 100
        token_sort = fuzz.token_sort_ratio(name1, name2)
        token_set = fuzz.token_set_ratio(name1, name2)
        
        # Address features
        addr_jaccard = compute_jaccard(addr1, addr2) * 100
        addr_ngram_sim = get_ngram_cos_sim(addr1, addr2) * 100
        
        # Numbers / zip
        zip_exact = 1.0 if (row['address_numbers_1'] and row['address_numbers_1'] == row['address_numbers_2']) else 0.0
        
        # Overlaps and lengths
        t1, t2 = row['name_tokens_1'], row['name_tokens_2']
        length_ratio = min(t1, t2) / max(t1, t2) if max(t1, t2) > 0 else 0.0
        
        set1 = set(name1.split())
        set2 = set(name2.split())
        name_overlap = len(set1.intersection(set2))
        
        features.append({
            'source1_entity_id': row['source1_entity_id'],
            'candidate_entity_id': row['candidate_entity_id'],
            'name_lev_ratio': lev_ratio,
            'name_jw_dist': jw_dist,
            'name_token_sort': token_sort,
            'name_token_set': token_set,
            'addr_jaccard': addr_jaccard,
            'addr_ngram_sim': addr_ngram_sim,
            'zip_exact': zip_exact,
            'name_length_ratio': length_ratio,
            'name_token_overlap': name_overlap
        })
        
    return pd.DataFrame(features)
