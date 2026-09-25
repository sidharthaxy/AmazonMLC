import os
import sys

# Ensure src can be imported whether PYTHONPATH is . or code directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(CURRENT_DIR)
for p in [CODE_DIR, os.path.join(os.getcwd(), 'aads_submission', 'business_entity_resolution', 'code')]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import argparse
import pandas as pd
import numpy as np
import time
import gc
from collections import defaultdict

from src.preprocessing import load_data, clean_text, normalize_abbreviations, extract_numerical_tokens
from src.candidate_generation import CompactInvertedIndex
from src.features import build_batch_features
from src.model import EntityMatchingModel
from src.evaluate import optimize_threshold, evaluate_macro_f05

def build_y_true_dict(df_gt: pd.DataFrame) -> dict:
    y_true = {}
    for _, row in df_gt.iterrows():
        s1_id = row['source1_entity_id']
        matches_str = str(row['matched_entity_ids']).strip()
        if not matches_str or matches_str.lower() == 'nan':
            y_true[s1_id] = set()
        else:
            y_true[s1_id] = set(matches_str.split(','))
    return y_true

def prepare_text_arrays(df: pd.DataFrame):
    """
    Extracts cleaned and normalized arrays for fast, memory-compact processing.
    """
    names_clean = [clean_text(t) for t in df['business_name'].values]
    names_norm = [normalize_abbreviations(t) for t in names_clean]
    addrs_clean = [clean_text(t) for t in df['business_address'].values]
    addrs_norm = [normalize_abbreviations(t) for t in addrs_clean]
    nums = [extract_numerical_tokens(t) for t in addrs_clean]
    return names_clean, names_norm, addrs_clean, addrs_norm, nums

def train_or_load_model(args, df_s1: pd.DataFrame, df_s2_s3: pd.DataFrame, model_path: str) -> tuple:
    """
    Loads an existing model or trains one on ground truth pairs if --is_train is set.
    Returns (model, best_threshold).
    """
    model = EntityMatchingModel()
    best_th = args.threshold
    
    # 1. If not training, attempt to load existing model
    if not args.is_train and os.path.exists(model_path):
        if model.load(model_path):
            print(f"Loaded trained model from {model_path}")
            return model, best_th
            
    # 2. Check if training is requested or needed
    gt_path = os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv')
    if (args.is_train or not os.path.exists(model_path)) and os.path.exists(gt_path):
        print("Training model on ground truth matching pairs...")
        df_gt = load_data(gt_path)
        gt_dict = build_y_true_dict(df_gt)
        del df_gt
        gc.collect()
        
        # Build positive and negative pairs from available data
        s23_map = {eid: i for i, eid in enumerate(df_s2_s3['entity_id'].values)}
        
        pos_q_idx = []
        pos_t_idx = []
        neg_q_idx = []
        neg_t_idx = []
        
        n_sample = min(20000, len(df_s1))
        for i in range(n_sample):
            s1_id = df_s1['entity_id'].iloc[i]
            if s1_id in gt_dict:
                matches = gt_dict[s1_id]
                for m in matches:
                    if m in s23_map:
                        pos_q_idx.append(i)
                        pos_t_idx.append(s23_map[m])
                        neg_q_idx.append(i)
                        neg_t_idx.append(np.random.randint(0, len(df_s2_s3)))
                        
        if pos_q_idx:
            q_indices = np.array(pos_q_idx + neg_q_idx, dtype=np.int32)
            t_indices = np.array(pos_t_idx + neg_t_idx, dtype=np.uint32)
            labels = np.array([1]*len(pos_q_idx) + [0]*len(neg_q_idx), dtype=np.int32)
            
            s1_sub = df_s1.iloc[:n_sample].reset_index(drop=True)
            s1_clean, s1_norm, s1_addrs_clean, s1_addrs_norm, s1_nums = prepare_text_arrays(s1_sub)
            s1_proc = pd.DataFrame({
                'entity_id': s1_sub['entity_id'].values,
                'business_name_norm': s1_norm,
                'business_address_norm': s1_addrs_norm,
                'address_numbers': s1_nums
            })
            
            t_unique_idx = np.unique(t_indices)
            t_unique_map = {orig_i: new_i for new_i, orig_i in enumerate(t_unique_idx)}
            target_sub = df_s2_s3.iloc[t_unique_idx].reset_index(drop=True)
            
            t_clean, t_norm, t_addrs_clean, t_addrs_norm, t_nums = prepare_text_arrays(target_sub)
            target_proc = pd.DataFrame({
                'entity_id': target_sub['entity_id'].values,
                'business_name_norm': t_norm,
                'business_address_norm': t_addrs_norm,
                'address_numbers': t_nums
            })
            
            remapped_t_idx = np.array([t_unique_map[ti] for ti in t_indices], dtype=np.uint32)
            
            df_features = build_batch_features(s1_proc, target_proc, q_indices, remapped_t_idx)
            df_features['label'] = labels
            
            print(f"Training XGBoost on {len(df_features)} pairs ({len(pos_q_idx)} positive, {len(neg_q_idx)} negative)...")
            model.fit(df_features, df_features['label'])
            model.save(model_path)
            print(f"Model successfully saved to {model_path}")
            
            del s1_proc, target_proc, df_features, s23_map, gt_dict
            gc.collect()
            return model, best_th
            
    print("Using high-precision composite heuristic / default model.")
    return model, best_th

def main(args):
    start_time = time.time()
    
    # 1. Load Data
    print("Loading data...")
    if args.is_train:
        s1_path = os.path.join(args.data_dir, 'train', 'train_source1.tsv')
        s2_path = os.path.join(args.data_dir, 'train', 'train_source2.tsv')
        s3_path = os.path.join(args.data_dir, 'train', 'train_source3.tsv')
    else:
        s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
        s2_test = os.path.join(args.data_dir, 'test', 'test_source2.tsv')
        s3_test = os.path.join(args.data_dir, 'test', 'test_source3.tsv')
        s2_path = s2_test if os.path.exists(s2_test) else os.path.join(args.data_dir, 'train', 'train_source2.tsv')
        s3_path = s3_test if os.path.exists(s3_test) else os.path.join(args.data_dir, 'train', 'train_source3.tsv')
        
    df_s1 = load_data(s1_path)
    df_s2 = load_data(s2_path)
    df_s3 = load_data(s3_path)
    
    df_s2_s3 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    
    if args.subset > 0:
        df_s1 = df_s1.head(args.subset)
        print(f"Using subset of {args.subset} S1 records for execution.")
        
    # Clean country strings for robust partitioning
    df_s1['country_clean'] = [clean_text(c) for c in df_s1['country'].values]
    df_s2_s3['country_clean'] = [clean_text(c) for c in df_s2_s3['country'].values]
    
    # 2. Model setup / training
    model, best_th = train_or_load_model(args, df_s1, df_s2_s3, args.model_path)
    if args.threshold is not None and not args.is_train:
        best_th = args.threshold
        
    print(f"Using decision threshold: {best_th:.4f}")
    
    # 3. Setup output file buffers
    os.makedirs(os.path.dirname(args.matching_out), exist_ok=True)
    if os.path.exists(args.matching_out):
        os.remove(args.matching_out)
        
    save_candidates = bool(args.candidate_out)
    if save_candidates:
        os.makedirs(os.path.dirname(args.candidate_out), exist_ok=True)
        if os.path.exists(args.candidate_out):
            os.remove(args.candidate_out)
            
    is_first_matching = True
    is_first_candidate = True
    
    # 4. Country Partitioning and Chunked Execution on S1
    unique_countries = sorted(list(set(df_s1['country_clean'].unique()).union(set(df_s2_s3['country_clean'].unique()))))
    total_processed_s1 = 0
    total_matches_found = 0
    
    for country in unique_countries:
        s1_country = df_s1[df_s1['country_clean'] == country].reset_index(drop=True)
        s2_s3_country = df_s2_s3[df_s2_s3['country_clean'] == country].reset_index(drop=True)
        
        if s1_country.empty:
            continue
            
        if s2_s3_country.empty:
            # All S1 entities in this country are singletons
            batch_results = pd.DataFrame({
                'source1_entity_id': s1_country['entity_id'].values,
                'matched_entity_ids': [''] * len(s1_country)
            })
            batch_results.to_csv(args.matching_out, sep='\t', index=False, header=is_first_matching, mode='a')
            is_first_matching = False
            total_processed_s1 += len(s1_country)
            del s1_country, s2_s3_country, batch_results
            gc.collect()
            continue
            
        print(f"\n--- Country: {country} | S1: {len(s1_country)} | S2+S3: {len(s2_s3_country)} ---")
        
        # Preprocess target pool text arrays
        t_clean, t_norm, t_addrs_clean, t_addrs_norm, t_nums = prepare_text_arrays(s2_s3_country)
        df_target_proc = pd.DataFrame({
            'entity_id': s2_s3_country['entity_id'].values,
            'business_name_norm': t_norm,
            'business_address_norm': t_addrs_norm,
            'address_numbers': t_nums
        })
        t_ids = s2_s3_country['entity_id'].values
        
        # Build memory-compact inverted index
        index = CompactInvertedIndex(max_block_size=10000, max_candidates=args.top_k)
        idx_t0 = time.time()
        index.build(t_clean, t_addrs_clean, t_nums)
        print(f"Built inverted index for '{country}' in {time.time() - idx_t0:.2f} s | Keys: {len(index.index)}")
        
        # Preprocess S1 text arrays
        q_clean, q_norm, q_addrs_clean, q_addrs_norm, q_nums = prepare_text_arrays(s1_country)
        df_s1_proc = pd.DataFrame({
            'entity_id': s1_country['entity_id'].values,
            'business_name_norm': q_norm,
            'business_address_norm': q_addrs_norm,
            'address_numbers': q_nums
        })
        
        num_batches = int(np.ceil(len(s1_country) / args.batch_size))
        print(f"Processing S1 in {num_batches} batches (batch_size={args.batch_size})...")
        
        for b in range(num_batches):
            b_t0 = time.time()
            b_start = b * args.batch_size
            b_end = min((b + 1) * args.batch_size, len(s1_country))
            
            sub_q_clean = q_clean[b_start:b_end]
            sub_q_addrs_clean = q_addrs_clean[b_start:b_end]
            sub_q_nums = q_nums[b_start:b_end]
            
            sub_s1_df = df_s1_proc.iloc[b_start:b_end].reset_index(drop=True)
            batch_s1_ids = sub_s1_df['entity_id'].values
            
            # Query candidate pairs for this batch only
            q_idx, t_idx = index.query_candidates(
                sub_q_clean, sub_q_addrs_clean, sub_q_nums, 
                max_candidates=args.top_k
            )
            
            # Optional streaming to candidate_pairs.tsv
            if save_candidates:
                cand_map = defaultdict(list)
                for qi, ti in zip(q_idx, t_idx):
                    cand_map[qi].append(t_ids[ti])
                    
                cand_rows = []
                for qi in range(len(batch_s1_ids)):
                    c_list = cand_map.get(qi, [])
                    cand_rows.append({
                        'source1_entity_id': batch_s1_ids[qi],
                        'candidate_entity_ids': ','.join(c_list) if c_list else ''
                    })
                df_cand_batch = pd.DataFrame(cand_rows)
                df_cand_batch.to_csv(args.candidate_out, sep='\t', index=False, header=is_first_candidate, mode='a')
                is_first_candidate = False
                del cand_map, cand_rows, df_cand_batch
                
            # Feature extraction and scoring
            matched_map = defaultdict(list)
            if len(q_idx) > 0:
                df_features = build_batch_features(sub_s1_df, df_target_proc, q_idx, t_idx)
                scores = model.predict_proba(df_features)
                df_features['score'] = scores
                
                # Filter surviving matches
                df_surviving = df_features[df_features['score'] >= best_th]
                for _, row in df_surviving.iterrows():
                    matched_map[row['source1_entity_id']].append(row['candidate_entity_id'])
                    
                del df_features, df_surviving, scores
                
            # Build batch results and stream to matching_results.tsv
            matching_rows = []
            for s1_id in batch_s1_ids:
                matches = matched_map.get(s1_id, [])
                match_str = ','.join(dict.fromkeys(matches)) if matches else ''
                matching_rows.append({
                    'source1_entity_id': s1_id,
                    'matched_entity_ids': match_str
                })
                if match_str:
                    total_matches_found += 1
                    
            df_match_batch = pd.DataFrame(matching_rows)
            df_match_batch.to_csv(args.matching_out, sep='\t', index=False, header=is_first_matching, mode='a')
            is_first_matching = False
            total_processed_s1 += len(batch_s1_ids)
            
            # Print batch progress
            print(f"  Batch {b+1}/{num_batches} ({len(batch_s1_ids)} records) | Candidates: {len(q_idx)} | Matches: {len(matched_map)} | Time: {time.time() - b_t0:.2f} s")
            
            # Immediate batch garbage collection
            del q_idx, t_idx, matched_map, matching_rows, df_match_batch, sub_s1_df
            gc.collect()
            
        # Free country memory
        del index, df_target_proc, df_s1_proc, s1_country, s2_s3_country
        gc.collect()
        
    print(f"\n==========================================")
    print(f"Pipeline finished successfully in {time.time() - start_time:.1f} seconds.")
    print(f"Total S1 entities processed: {total_processed_s1}")
    print(f"Entities with predicted matches: {total_matches_found}")
    print(f"Matching results saved to: {args.matching_out}")
    if save_candidates:
        print(f"Candidate pairs saved to: {args.candidate_out}")
    print(f"==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--is_train', action='store_true', help='Run in training mode')
    parser.add_argument('--candidate_out', type=str, default='output/candidate_pairs.tsv', help='Candidate pairs TSV output path')
    parser.add_argument('--matching_out', type=str, default='output/matching_results.tsv', help='Matching results TSV output path')
    parser.add_argument('--model_path', type=str, default='models/entity_model.json', help='Model checkpoint path')
    parser.add_argument('--batch_size', type=int, default=50000, help='Batch size for S1 chunking')
    parser.add_argument('--top_k', type=int, default=30, help='Max candidates per query record')
    parser.add_argument('--threshold', type=float, default=0.55, help='Score decision threshold')
    parser.add_argument('--subset', type=int, default=0, help='Subset size for fast baseline execution (0 for full)')
    args = parser.parse_args()
    main(args)
