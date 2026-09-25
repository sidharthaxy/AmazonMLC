import pandas as pd
import numpy as np
from rapidfuzz import fuzz, distance
from typing import Sequence

def compute_jaccard(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return float(intersection / union) if union > 0 else 0.0

def get_ngram_cos_sim(s1: str, s2: str, n: int = 3) -> float:
    if not s1 or not s2:
        return 0.0
    
    len1 = len(s1)
    len2 = len(s2)
    if len1 < n or len2 < n:
        return float(s1 == s2)
        
    set1 = {s1[i:i+n] for i in range(len1 - n + 1)}
    set2 = {s2[i:i+n] for i in range(len2 - n + 1)}
    
    if not set1 or not set2:
        return 0.0
        
    intersection = len(set1.intersection(set2))
    norm = np.sqrt(len(set1)) * np.sqrt(len(set2))
    return float(intersection / norm) if norm > 0 else 0.0

def build_batch_features(
    df_query: pd.DataFrame, 
    df_target: pd.DataFrame, 
    query_indices: np.ndarray, 
    target_indices: np.ndarray
) -> pd.DataFrame:
    """
    Computes pairwise similarity features between df_query and df_target using integer index arrays.
    Avoids expensive DataFrame merges and large intermediate allocations.
    """
    if len(query_indices) == 0:
        return pd.DataFrame()
        
    # Extract string columns directly
    q_name_col = 'business_name_norm' if 'business_name_norm' in df_query.columns else 'business_name_clean' if 'business_name_clean' in df_query.columns else 'business_name'
    t_name_col = 'business_name_norm' if 'business_name_norm' in df_target.columns else 'business_name_clean' if 'business_name_clean' in df_target.columns else 'business_name'
    
    q_addr_col = 'business_address_norm' if 'business_address_norm' in df_query.columns else 'business_address_clean' if 'business_address_clean' in df_query.columns else 'business_address'
    t_addr_col = 'business_address_norm' if 'business_address_norm' in df_target.columns else 'business_address_clean' if 'business_address_clean' in df_target.columns else 'business_address'
    
    has_q_nums = 'address_numbers' in df_query.columns
    has_t_nums = 'address_numbers' in df_target.columns
    
    q_names = df_query[q_name_col].values[query_indices]
    t_names = df_target[t_name_col].values[target_indices]
    
    q_addrs = df_query[q_addr_col].values[query_indices]
    t_addrs = df_target[t_addr_col].values[target_indices]
    
    q_nums = df_query['address_numbers'].values[query_indices] if has_q_nums else [""] * len(query_indices)
    t_nums = df_target['address_numbers'].values[target_indices] if has_t_nums else [""] * len(target_indices)
    
    q_ids = df_query['entity_id'].values[query_indices]
    t_ids = df_target['entity_id'].values[target_indices]
    
    # Calculate features in list comprehensions
    lev_ratios = [fuzz.ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    jw_dists = [distance.JaroWinkler.normalized_similarity(n1, n2) * 100 for n1, n2 in zip(q_names, t_names)]
    token_sorts = [fuzz.token_sort_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    token_sets = [fuzz.token_set_ratio(n1, n2) for n1, n2 in zip(q_names, t_names)]
    
    addr_jaccards = [compute_jaccard(a1, a2) * 100 for a1, a2 in zip(q_addrs, t_addrs)]
    addr_ngrams = [get_ngram_cos_sim(a1, a2) * 100 for a1, a2 in zip(q_addrs, t_addrs)]
    
    zip_exacts = [1.0 if (z1 and z1 == z2) else 0.0 for z1, z2 in zip(q_nums, t_nums)]
    
    token_lens_1 = [len(n.split()) for n in q_names]
    token_lens_2 = [len(n.split()) for n in t_names]
    length_ratios = [
        min(t1, t2) / max(t1, t2) if max(t1, t2) > 0 else 0.0 
        for t1, t2 in zip(token_lens_1, token_lens_2)
    ]
    
    name_overlaps = [
        float(len(set(n1.split()).intersection(set(n2.split()))))
        for n1, n2 in zip(q_names, t_names)
    ]
    
    return pd.DataFrame({
        'source1_entity_id': q_ids,
        'candidate_entity_id': t_ids,
        'name_lev_ratio': np.array(lev_ratios, dtype=np.float32),
        'name_jw_dist': np.array(jw_dists, dtype=np.float32),
        'name_token_sort': np.array(token_sorts, dtype=np.float32),
        'name_token_set': np.array(token_sets, dtype=np.float32),
        'addr_jaccard': np.array(addr_jaccards, dtype=np.float32),
        'addr_ngram_sim': np.array(addr_ngrams, dtype=np.float32),
        'zip_exact': np.array(zip_exacts, dtype=np.float32),
        'name_length_ratio': np.array(length_ratios, dtype=np.float32),
        'name_token_overlap': np.array(name_overlaps, dtype=np.float32),
    })

def build_feature_dataset(
    df_candidates: pd.DataFrame, 
    df_s1: pd.DataFrame, 
    df_s2_s3: pd.DataFrame
) -> pd.DataFrame:
    """
    Backward-compatible entry point that converts candidate pairs into features.
    """
    if df_candidates.empty:
        return pd.DataFrame()
        
    s1_id_to_idx = {eid: idx for idx, eid in enumerate(df_s1['entity_id'].values)}
    t_id_to_idx = {eid: idx for idx, eid in enumerate(df_s2_s3['entity_id'].values)}
    
    q_indices = []
    t_indices = []
    
    for _, row in df_candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(',') if row['candidate_entity_ids'] else []
        if s1_id in s1_id_to_idx:
            q_i = s1_id_to_idx[s1_id]
            for cand in cands:
                cand = cand.strip()
                if cand and cand in t_id_to_idx:
                    q_indices.append(q_i)
                    t_indices.append(t_id_to_idx[cand])
                    
    if not q_indices:
        return pd.DataFrame()
        
    return build_batch_features(
        df_s1, df_s2_s3, 
        np.array(q_indices, dtype=np.int32), 
        np.array(t_indices, dtype=np.uint32)
    )
