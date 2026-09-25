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
            scale_pos_weight=1.0 # Alternative to class_weight='balanced'
        )
        self.use_transformer = use_transformer
        self.transformer_model = None
        
        if self.use_transformer:
            from sentence_transformers import SentenceTransformer
            self.transformer_model = SentenceTransformer('intfloat/multilingual-e5-base')
            
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
        
    def predict_proba(self, df_features: pd.DataFrame) -> np.ndarray:
        X = self.prepare_X(df_features)
        if isinstance(self.model, xgb.Booster):
            return self.model.predict(xgb.DMatrix(X))
        else:
            return self.model.predict_proba(X)[:, 1]
        
    def save(self, path: str):
        self.model.save_model(path)
        
    def load(self, path: str):
        self.model = xgb.Booster()
        self.model.load_model(path)
