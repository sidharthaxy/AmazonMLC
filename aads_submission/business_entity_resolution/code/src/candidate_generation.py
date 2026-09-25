import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Tuple, Sequence, Dict, Set
import gc

GENERIC_STOP_WORDS: Set[str] = {
    'inc', 'llc', 'ltd', 'corp', 'company', 'corporation', 'limited',
    'pvt', 'private', 'the', 'and', 'street', 'road', 'avenue', 'drive',
    'lane', 'dr', 'st', 'rd', 'ave', 'blvd', 'suite', 'floor', 'unit',
    'center', 'services', 'service', 'enterprise', 'enterprises',
    'solutions', 'group', 'holdings', 'international', 'industries',
    'store', 'shop', 'market', 'restaurant', 'cafe', 'hotel'
}

def get_first_token(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    return tokens[0] if tokens else ""

class CompactInvertedIndex:
    """
    Memory-compact inverted index for candidate generation in entity resolution.
    Stores posting lists as 32-bit unsigned integer arrays (np.uint32) to minimize
    memory footprint on large datasets (10M+ rows).
    
    Automatically prunes high-frequency keys (> max_block_size) to avoid
    quadratic candidate blowups and OOM.
    """
    def __init__(self, max_block_size: int = 10000, max_candidates: int = 30):
        self.max_block_size = max_block_size
        self.max_candidates = max_candidates
        self.index: Dict[str, np.ndarray] = {}
        self.num_records = 0

    @staticmethod
    def extract_keys(name: str, addr: str, nums: str) -> List[str]:
        keys = []
        if name:
            name_len = len(name)
            if name_len >= 4:
                keys.append('p4:' + name[:4])
            if name_len >= 3:
                keys.append('p3:' + name[:3])
            if 3 <= name_len <= 25:
                keys.append('exact:' + name)
                
            words = [w for w in name.split() if w not in GENERIC_STOP_WORDS and len(w) >= 3]
            if words:
                keys.append('w1:' + words[0])
                if len(words) > 1:
                    keys.append('w2:' + words[1])
                if len(words) > 2:
                    keys.append('wlast:' + words[-1])
                    
        num_list = nums.split() if nums else []
        addr_words = [w for w in addr.split() if w not in GENERIC_STOP_WORDS and len(w) >= 4] if addr else []
        
        if num_list:
            for num in num_list:
                if len(num) >= 4:
                    keys.append('num:' + num)
            if addr_words:
                keys.append('num_w:' + num_list[0] + '_' + addr_words[0])
            if name and len(name) >= 3:
                keys.append('num_p3:' + num_list[0] + '_' + name[:3])
                
        if addr_words:
            keys.append('aw1:' + addr_words[0])
            if len(addr_words) > 1:
                keys.append('aw2:' + addr_words[1])
                
        return keys

    def build(self, names: Sequence[str], addrs: Sequence[str], nums: Sequence[str]):
        """
        Builds the inverted index from target pool arrays.
        """
        self.num_records = len(names)
        raw_index = defaultdict(list)
        
        for i in range(self.num_records):
            keys = self.extract_keys(names[i], addrs[i], nums[i])
            for k in keys:
                raw_index[k].append(i)
                
        # Compaction: drop blocks > max_block_size, convert retained to np.uint32
        self.index = {}
        for k, v in raw_index.items():
            if len(v) <= self.max_block_size:
                self.index[k] = np.array(v, dtype=np.uint32)
                
        del raw_index
        gc.collect()

    def query_candidates(
        self, 
        query_names: Sequence[str], 
        query_addrs: Sequence[str], 
        query_nums: Sequence[str],
        max_candidates: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Queries the inverted index for a batch of query records.
        Returns:
            query_indices: 1D array of row indices in the query batch
            target_indices: 1D array of row indices in the target index
        """
        if max_candidates is None:
            max_candidates = self.max_candidates
            
        all_query_idx = []
        all_target_idx = []
        
        index_map = self.index
        n_queries = len(query_names)
        
        for q_i in range(n_queries):
            keys = self.extract_keys(query_names[q_i], query_addrs[q_i], query_nums[q_i])
            if not keys:
                continue
                
            cand_counts = Counter()
            for k in keys:
                if k in index_map:
                    cand_counts.update(index_map[k])
                    
            if not cand_counts:
                continue
                
            if len(cand_counts) > max_candidates:
                top_cands = [c for c, _ in cand_counts.most_common(max_candidates)]
            else:
                top_cands = list(cand_counts.keys())
                
            for t_i in top_cands:
                all_query_idx.append(q_i)
                all_target_idx.append(t_i)
                
        return np.array(all_query_idx, dtype=np.int32), np.array(all_target_idx, dtype=np.uint32)

def generate_candidates(
    df_s1: pd.DataFrame, 
    df_s2_s3: pd.DataFrame, 
    top_k: int = 30, 
    batch_size: int = 50000
) -> pd.DataFrame:
    """
    Generates candidate pairs using country partitioning and memory-compact inverted index.
    Backward-compatible with original API returning DataFrame of:
    ['source1_entity_id', 'candidate_entity_ids']
    """
    candidate_records = []
    
    # Ensure country column exists
    s1_countries = df_s1['country_clean'].values if 'country_clean' in df_s1.columns else df_s1['country'].str.strip().str.lower().values
    s23_countries = df_s2_s3['country_clean'].values if 'country_clean' in df_s2_s3.columns else df_s2_s3['country'].str.strip().str.lower().values
    
    unique_countries = set(s1_countries).union(set(s23_countries))
    
    for country in unique_countries:
        s1_mask = (s1_countries == country)
        s23_mask = (s23_countries == country)
        
        s1_country = df_s1[s1_mask].reset_index(drop=True)
        s23_country = df_s2_s3[s23_mask].reset_index(drop=True)
        
        if s1_country.empty:
            continue
            
        if s23_country.empty:
            for s1_id in s1_country['entity_id'].values:
                candidate_records.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ''})
            continue
            
        print(f"Candidate generation for country: {country} | S1: {len(s1_country)} | S2+S3: {len(s23_country)}")
        
        # Prepare target text arrays
        t_names = s23_country['business_name_norm'].values if 'business_name_norm' in s23_country.columns else s23_country['business_name'].values
        t_addrs = s23_country['business_address_norm'].values if 'business_address_norm' in s23_country.columns else s23_country['business_address'].values
        t_nums = s23_country['address_numbers'].values if 'address_numbers' in s23_country.columns else [""] * len(s23_country)
        t_ids = s23_country['entity_id'].values
        
        # Build inverted index for this country
        index = CompactInvertedIndex(max_block_size=10000, max_candidates=top_k)
        index.build(t_names, t_addrs, t_nums)
        
        # Query in batches
        num_batches = int(np.ceil(len(s1_country) / batch_size))
        for b in range(num_batches):
            b_start = b * batch_size
            b_end = min((b + 1) * batch_size, len(s1_country))
            s1_batch = s1_country.iloc[b_start:b_end]
            
            q_names = s1_batch['business_name_norm'].values if 'business_name_norm' in s1_batch.columns else s1_batch['business_name'].values
            q_addrs = s1_batch['business_address_norm'].values if 'business_address_norm' in s1_batch.columns else s1_batch['business_address'].values
            q_nums = s1_batch['address_numbers'].values if 'address_numbers' in s1_batch.columns else [""] * len(s1_batch)
            q_ids = s1_batch['entity_id'].values
            
            q_idx, t_idx = index.query_candidates(q_names, q_addrs, q_nums, max_candidates=top_k)
            
            # Group candidates by query index
            batch_cand_map = defaultdict(list)
            for qi, ti in zip(q_idx, t_idx):
                batch_cand_map[qi].append(t_ids[ti])
                
            for qi in range(len(s1_batch)):
                cands = batch_cand_map.get(qi, [])
                cand_str = ",".join(cands) if cands else ""
                candidate_records.append({'source1_entity_id': q_ids[qi], 'candidate_entity_ids': cand_str})
                
            del q_idx, t_idx, batch_cand_map
            gc.collect()
            
        del index, s23_country, s1_country
        gc.collect()
        
    return pd.DataFrame(candidate_records)
