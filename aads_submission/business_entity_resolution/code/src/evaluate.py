import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def evaluate_macro_f05(y_true: Dict[str, set], y_pred: Dict[str, set]) -> float:
    """
    Computes macro-averaged F_0.5 score across all Source 1 entities.
    y_true: dict mapping Source 1 entity_id to a set of true matching entity_ids.
    y_pred: dict mapping Source 1 entity_id to a set of predicted matching entity_ids.
    """
    f05_scores = []
    
    for s1_id in y_true.keys():
        true_matches = y_true[s1_id]
        pred_matches = y_pred.get(s1_id, set())
        
        # Singleton logic
        if len(true_matches) == 0:
            if len(pred_matches) == 0:
                f05_scores.append(1.0)
            else:
                f05_scores.append(0.0)
            continue
            
        if len(pred_matches) == 0:
            f05_scores.append(0.0)
            continue
            
        true_positives = len(true_matches.intersection(pred_matches))
        
        precision = true_positives / len(pred_matches) if len(pred_matches) > 0 else 0.0
        recall = true_positives / len(true_matches) if len(true_matches) > 0 else 0.0
        
        if precision == 0 and recall == 0:
            f05 = 0.0
        else:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
            
        f05_scores.append(f05)
        
    return np.mean(f05_scores) if f05_scores else 0.0

def optimize_threshold(df_scores: pd.DataFrame, y_true: Dict[str, set], thresholds: List[float] = None) -> Tuple[float, float]:
    """
    Finds the optimal probability threshold to maximize macro F_0.5.
    df_scores should have: ['source1_entity_id', 'candidate_entity_id', 'score']
    y_true: true matches dict mapping s1_id -> set(matches)
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)
        
    best_threshold = 0.5
    best_f05 = -1.0
    
    for th in thresholds:
        # Generate predictions for this threshold
        df_pred = df_scores[df_scores['score'] >= th]
        
        y_pred = {}
        for s1_id in y_true.keys():
            y_pred[s1_id] = set()
            
        # Group by S1
        grouped = df_pred.groupby('source1_entity_id')['candidate_entity_id'].apply(set).to_dict()
        for s1_id, cands in grouped.items():
            if s1_id in y_pred:
                y_pred[s1_id] = cands
                
        f05 = evaluate_macro_f05(y_true, y_pred)
        print(f"Threshold: {th:.2f} | Macro F0.5: {f05:.4f}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = th
            
    return best_threshold, best_f05
