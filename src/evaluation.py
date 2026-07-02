import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from sklearn.model_selection import train_test_split
from src.hybrid_ranker import HybridRanker

logger = logging.getLogger(__name__)

def compute_ndcg(
    ranked_df: pd.DataFrame,
    labels_path: str = 'data/processed/relevance_labels.csv',
    k_values: list = None
) -> dict:
    """
    Computes NDCG@K for the ranked candidates using stored human labels.
    """
    if k_values is None:
        k_values = [5, 10]
        
    if not os.path.exists(labels_path):
        logger.warning(f"Labels file not found at {labels_path}")
        return {}
        
    labels_df = pd.read_csv(labels_path)
    
    # Map candidate_id to relevance (taking the max relevance if there are duplicate labels)
    relevance_map = labels_df.groupby('candidate_id')['relevance'].max().to_dict()
    string_relevance_map = {str(k): v for k, v in relevance_map.items()}
    
    # Count how many of the ranked candidates are actually in the labeled set
    ranked_ids = ranked_df['candidate_id'].astype(str).tolist()
    matches = sum(1 for cid in ranked_ids if cid in string_relevance_map)
    
    if matches < 5:
        print("Too few labeled candidates for reliable NDCG. Label more candidates first with label_candidates.py")
        return {}
        
    # Build true_relevance and predicted scores in the SAME ORDER as ranked_df
    # If not in labels -> assume relevance=0 (conservative)
    true_relevance = np.array([string_relevance_map.get(cid, 0) for cid in ranked_ids])
    predicted_scores = ranked_df['final_score'].fillna(0.0).values
    
    # sklearn ndcg_score expects 2D arrays: shape (n_queries, n_items). 
    # Since we have exactly 1 query (the JD), shape is (1, n_items).
    y_true = np.asarray([true_relevance])
    y_score = np.asarray([predicted_scores])
    
    results = {}
    # print(f"{'Metric':<12} {'Score'}")
    
    for k in k_values:
        actual_k = min(k, len(ranked_df))
        score = ndcg_score(y_true, y_score, k=actual_k)
        results[f'ndcg@{k}'] = score
        # print(f"NDCG@{k:<7} {score:.4f}")
        
    return results

def compare_rankers(candidates: list[dict], jd_text: str) -> None:
    """
    Runs HybridRanker with and without LTR and prints a side-by-side NDCG comparison,
    ensuring a proper train/test split to avoid label leakage. Runs 5 random splits.
    """
    labels_path = 'data/processed/relevance_labels.csv'
    if not os.path.exists(labels_path):
        print(f"Labels file not found at {labels_path}")
        return
        
    labels_df = pd.read_csv(labels_path)
    if len(labels_df) < 10:
        print("Not enough labels for train/test split.")
        return
        
    ndcg_weighted_scores = []
    ndcg_ltr_scores = []
    
    print("Running Weighted Sum baseline (on all candidates)...")
    ranker = HybridRanker()
    df_weighted = ranker.rank(candidates, jd_text, use_ltr=False)
    
    num_iterations = 20
    for i in range(num_iterations):
        print(f"\n--- Split {i+1} of {num_iterations} ---")
        train_labels, test_labels = train_test_split(labels_df, test_size=0.2, random_state=42 + i)
        
        # Temporarily save train/test splits for the ranker to use during evaluation
        train_labels.to_csv('data/processed/temp_train_labels.csv', index=False)
        test_labels.to_csv('data/processed/temp_test_labels.csv', index=False)
        
        print(f"Training LightGBM LTR on Train Set & Predicting (Split {i+1})...")
        # Temporarily rename the labels file so HybridRanker uses the train set for training
        os.rename('data/processed/relevance_labels.csv', 'data/processed/relevance_labels_backup.csv')
        os.rename('data/processed/temp_train_labels.csv', 'data/processed/relevance_labels.csv')
        
        # Also remove the existing model so it is forced to re-train on the temp train set
        model_path = 'data/processed/ltr_model.pkl'
        model_backup = 'data/processed/ltr_model_backup.pkl'
        if os.path.exists(model_path):
            os.rename(model_path, model_backup)
            
        try:
            df_ltr = ranker.rank(candidates, jd_text, use_ltr=True)
        finally:
            # Clean up and restore original files
            os.remove('data/processed/relevance_labels.csv')
            os.rename('data/processed/relevance_labels_backup.csv', 'data/processed/relevance_labels.csv')
            
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(model_backup):
                os.rename(model_backup, model_path)
                
        # print(f"Computing TRUE NDCG for Weighted Sum (Split {i+1}):")
        ndcg_w = compute_ndcg(df_weighted, labels_path='data/processed/temp_test_labels.csv')
        
        # print(f"Computing TRUE NDCG for LightGBM LTR (Split {i+1}):")
        ndcg_l = compute_ndcg(df_ltr, labels_path='data/processed/temp_test_labels.csv')
        
        if ndcg_w and ndcg_l:
            ndcg_weighted_scores.append(ndcg_w)
            ndcg_ltr_scores.append(ndcg_l)
        
        # Clean up temp test labels after split is complete
        if os.path.exists('data/processed/temp_test_labels.csv'):
            os.remove('data/processed/temp_test_labels.csv')
            
    if not ndcg_weighted_scores or not ndcg_ltr_scores:
        print("\nCould not generate comparison due to missing NDCG scores.")
        return
        
    # Average the results over the iterations
    avg_ndcg_weighted = {k: np.mean([d[k] for d in ndcg_weighted_scores]) for k in ndcg_weighted_scores[0]}
    avg_ndcg_ltr = {k: np.mean([d[k] for d in ndcg_ltr_scores]) for k in ndcg_ltr_scores[0]}

    print("\n" + "=" * 55)
    print(f"{num_iterations}-FOLD REPEATED RANDOM SUBSAMPLING TRUE NDCG RESULTS")
    print(f"{'Metric':<12} {'Weighted Sum':<15} {'LightGBM LTR':<15} {'Delta'}")
    print("-" * 55)

    for metric in avg_ndcg_weighted.keys():
        score_w = avg_ndcg_weighted[metric]
        score_l = avg_ndcg_ltr[metric]
        delta = score_l - score_w
        
        # Format delta cleanly
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        metric_display = metric.upper()
        
        print(f"{metric_display:<12} {score_w:<15.4f} {score_l:<15.4f} {delta_str}")
    print("=" * 55)

if __name__ == '__main__':
    from src.data_loader import load_candidates, load_job_description
    candidates = load_candidates('data/raw/sample_candidates.json')
    jd_text = load_job_description('data/raw/job_description.docx')
    compare_rankers(candidates, jd_text)
