import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def get_first_token(text: str) -> str:
    if not text:
        return ""
    tokens = text.split()
    return tokens[0] if tokens else ""

def generate_candidates(df_s1: pd.DataFrame, df_s2_s3: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """
    Generates candidate pairs using blocking strategies partitioned by country.
    df_s2_s3 is a combined dataframe of Source 2 and Source 3.
    """
    candidate_pairs = []
    
    # Ensure columns exist
    df_s1['first_token'] = df_s1['business_name_clean'].apply(get_first_token)
    df_s2_s3['first_token'] = df_s2_s3['business_name_clean'].apply(get_first_token)
    
    countries = set(df_s1['country_clean'].unique()).union(set(df_s2_s3['country_clean'].unique()))
    
    for country in countries:
        # Filter by country
        s1_country = df_s1[df_s1['country_clean'] == country].reset_index(drop=True)
        s2_s3_country = df_s2_s3[df_s2_s3['country_clean'] == country].reset_index(drop=True)
        
        if s1_country.empty or s2_s3_country.empty:
            continue
            
        print(f"Blocking for country: {country} | S1: {len(s1_country)} | S2+S3: {len(s2_s3_country)}")
        
        # 1. Lexical Blocking using TF-IDF + Cosine Similarity
        # Combine name and address for better text representation
        s1_text = s1_country['business_name_norm'] + " " + s1_country['business_address_norm']
        s2_s3_text = s2_s3_country['business_name_norm'] + " " + s2_s3_country['business_address_norm']
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=2, max_df=0.8)
        
        # Fit on both to ensure same vocabulary
        vectorizer.fit(pd.concat([s1_text, s2_s3_text]))
        
        s1_tfidf = vectorizer.transform(s1_text)
        s2_s3_tfidf = vectorizer.transform(s2_s3_text)
        
        # Determine actual k (cannot be greater than the number of candidates)
        actual_k = min(top_k, s2_s3_tfidf.shape[0])
        
        # NearestNeighbors for cosine similarity
        nn = NearestNeighbors(n_neighbors=actual_k, metric='cosine', n_jobs=-1, algorithm='brute')
        nn.fit(s2_s3_tfidf)
        
        distances, indices = nn.kneighbors(s1_tfidf)
        
        # Collect TF-IDF candidates
        s2_s3_ids = s2_s3_country['entity_id'].values
        s1_ids = s1_country['entity_id'].values
        
        country_candidates = {s1_id: set() for s1_id in s1_ids}
        
        for i, s1_id in enumerate(s1_ids):
            for j in range(actual_k):
                # distance is 1 - cosine_similarity. 
                # Optional: filter by distance threshold to reduce candidates, e.g., if distance < 0.6
                if distances[i, j] < 0.7:  
                    country_candidates[s1_id].add(s2_s3_ids[indices[i, j]])
                    
        # 2. Rule-based Blocking: First token match
        s2_s3_token_map = s2_s3_country.groupby('first_token')['entity_id'].apply(list).to_dict()
        for i, row in s1_country.iterrows():
            token = row['first_token']
            if token and len(token) > 2: # Avoid matching on very short tokens like 'a'
                matches = s2_s3_token_map.get(token, [])
                # Add up to some limit to avoid explosion if a token is very common
                country_candidates[row['entity_id']].update(matches[:50])
                
        # 3. Rule-based Blocking: Address numbers match (proxy for zip code)
        s2_s3_num_map = s2_s3_country[s2_s3_country['address_numbers'] != ''].groupby('address_numbers')['entity_id'].apply(list).to_dict()
        for i, row in s1_country.iterrows():
            nums = row['address_numbers']
            if nums and len(nums) > 3: # Require at least 4 digits to avoid common building numbers
                matches = s2_s3_num_map.get(nums, [])
                country_candidates[row['entity_id']].update(matches[:20])
                
        # Convert to candidate pairs dataframe
        for s1_id, cands in country_candidates.items():
            if cands:
                cand_str = ",".join(list(cands))
                candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': cand_str})
            else:
                candidate_pairs.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ""})
                
    # If there are S1 entities with no country matched or empty
    all_s1 = set(df_s1['entity_id'])
    processed_s1 = set([p['source1_entity_id'] for p in candidate_pairs])
    missing = all_s1 - processed_s1
    for m in missing:
        candidate_pairs.append({'source1_entity_id': m, 'candidate_entity_ids': ""})
        
    return pd.DataFrame(candidate_pairs)
