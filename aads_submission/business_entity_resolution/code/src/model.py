import os
import xgboost as xgb
import pandas as pd
import numpy as np

class EntityMatchingModel:
    def __init__(self, use_transformer=False):
        self.model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5, 
            random_state=42,
            scale_pos_weight=1.0,
            tree_method='hist',
            eval_metric='logloss'
        )
        self.use_transformer = use_transformer
        self.transformer_model = None
        self.is_fitted = False
        
        if self.use_transformer:
            try:
                from sentence_transformers import SentenceTransformer
                self.transformer_model = SentenceTransformer('intfloat/multilingual-e5-base')
            except Exception:
                self.transformer_model = None
            
    def prepare_X(self, df_features: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'name_lev_ratio', 'name_jw_dist', 'name_token_sort', 'name_token_set',
            'addr_jaccard', 'addr_ngram_sim', 'zip_exact',
            'name_length_ratio', 'name_token_overlap'
        ]
        if self.use_transformer and 'semantic_sim' in df_features.columns:
            feature_cols.append('semantic_sim')
        return df_features[feature_cols]

    def fit(self, df_features: pd.DataFrame, y: pd.Series):
        X = self.prepare_X(df_features)
        self.model.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        if df_features.empty:
            return np.array([], dtype=np.float32)
            
        X = self.prepare_X(df_features)
        if self.is_fitted:
            if isinstance(self.model, xgb.Booster):
                return self.model.predict(xgb.DMatrix(X)).astype(np.float32)
            else:
                return self.model.predict_proba(X)[:, 1].astype(np.float32)
        else:
            # Fallback high-precision composite scoring heuristic
            jw = X['name_jw_dist'].values / 100.0
            ts = X['name_token_set'].values / 100.0
            aj = X['addr_jaccard'].values / 100.0
            ze = X['zip_exact'].values
            score = 0.35 * jw + 0.25 * ts + 0.25 * aj + 0.15 * ze
            return score.astype(np.float32)
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_model(path)
        
    def load(self, path: str) -> bool:
        if os.path.exists(path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(path)
            self.is_fitted = True
            return True
        return False
