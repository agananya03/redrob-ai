import os
import pickle
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'skill_match_score',
    'experience_score',
    'education_score',
    'trajectory_score',
    'platform_signal_score',
    'semantic_score'
]

def load_training_data(labels_path: str, scores_df: pd.DataFrame):
    """
    Load relevance_labels.csv and join with scores_df to create features (X) and labels (y).
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
        
    labels_df = pd.read_csv(labels_path)
    
    if len(labels_df) < 10:
        raise ValueError("Need at least 10 labeled candidates to train. Run scripts/label_candidates.py first.")
        
    merged_df = pd.merge(labels_df, scores_df, on='candidate_id', how='inner')
    
    missing_count = len(labels_df) - len(merged_df)
    if missing_count > 0:
        logger.warning(f"Dropped {missing_count} labeled candidates not found in scores_df.")
        
    # Ensure we still have at least 10 after merging
    if len(merged_df) < 10:
        raise ValueError("Need at least 10 valid labeled candidates present in scores_df to train.")
        
    # X = the 6 features as a numpy array
    X = merged_df[FEATURE_NAMES].values
    
    # y = relevance column as integer array
    y = merged_df['relevance'].values.astype(int)
    
    print(f"load_training_data: X shape = {X.shape}, y shape = {y.shape}")
    print(f"load_training_data: First 3 rows of X = \n{X[:3]}")
    
    return X, y, FEATURE_NAMES


def train_ranker(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> lgb.Booster:
    """
    Trains a LightGBM LambdaRank model.
    """
    if os.path.exists('data/processed/ltr_model.pkl'):
        os.remove('data/processed/ltr_model.pkl')
        
    # Do an 80/20 train/val split. 
    # Note: with a small labeled set this is a sanity check, not a robust holdout.
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    # Since we have one JD (one query group), set group=[len(X)] for the respective splits
    train_data = lgb.Dataset(X_train, label=y_train, group=[len(X_train)], feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, group=[len(X_val)], feature_name=feature_names, reference=train_data)
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': 0.05,
        'min_data_in_leaf': min(5, len(X_train) // 2),  # Keep low for small datasets
        'verbose': -1
    }
    
    logger.info("Training LightGBM ranker...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, val_data]
    )
    
    # Print feature importances sorted descending
    importances = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values(by='importance', ascending=False)
    
    print("\nFeature Importances (Gain):")
    for _, row in imp_df.iterrows():
        print(f"  {row['feature']:<25} {row['importance']:.4f}")
        
    # Save a bar chart of feature importances
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.gca().invert_yaxis()  # Highest importance at the top
    plt.title('LightGBM Feature Importances (Gain)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig('outputs/feature_importance.png')
    plt.close()
    
    return model


def predict_scores(model: lgb.Booster, scores_df: pd.DataFrame) -> pd.Series:
    """
    Takes the trained model and the FULL scores_df and returns a pd.Series of predicted scores.
    """
    # Build X_all from the same 6 features
    X_all = scores_df[FEATURE_NAMES].fillna(0.0).values
    print(f"predict_scores: X_all shape = {X_all.shape}")
    print(f"predict_scores: First 3 rows of X_all = \n{X_all[:3]}")
    
    preds = model.predict(X_all)
    
    if np.isnan(preds).any():
        logger.warning("LightGBM predicted NaN scores! Replacing with 0.0.")
        preds = np.nan_to_num(preds, nan=0.0)
    
    # Min-Max Scale to 0-1 range to match legacy final_score output expectations
    if len(preds) > 1 and preds.max() > preds.min():
        preds = (preds - preds.min()) / (preds.max() - preds.min())
        
    return pd.Series(preds, index=scores_df['candidate_id'], name='ltr_score')


def save_model(model: lgb.Booster, path: str = 'data/processed/ltr_model.pkl'):
    """
    Saves the LightGBM model to disk via pickle.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path: str = 'data/processed/ltr_model.pkl') -> lgb.Booster:
    """
    Loads the LightGBM model from disk via pickle.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model
