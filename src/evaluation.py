import os
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
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
        k_values = [10, 50, 100]
        
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
    print(f"{'Metric':<12} {'Score'}")
    
    for k in k_values:
        actual_k = min(k, len(ranked_df))
        score = ndcg_score(y_true, y_score, k=actual_k)
        results[f'ndcg@{k}'] = score
        print(f"NDCG@{k:<7} {score:.4f}")
        
    return results

def compare_rankers(candidates: list[dict], jd_text: str) -> None:
    """
    Runs HybridRanker with and without LTR and prints a side-by-side NDCG comparison.
    """
    print("Running Weighted Sum baseline...")
    ranker = HybridRanker()
    
    # 1. Run Weighted Sum
    df_weighted = ranker.rank(candidates, jd_text, use_ltr=False)
    
    print("\nRunning LightGBM LTR...")
    # 2. Run LightGBM LTR
    df_ltr = ranker.rank(candidates, jd_text, use_ltr=True)
    
    print("\nComputing NDCG for Weighted Sum:")
    ndcg_weighted = compute_ndcg(df_weighted)
    
    print("\nComputing NDCG for LightGBM LTR:")
    ndcg_ltr = compute_ndcg(df_ltr)
    
    if not ndcg_weighted or not ndcg_ltr:
        print("\nCould not generate comparison due to missing NDCG scores.")
        return
        
    print("\n" + "=" * 55)
    print(f"{'Metric':<12} {'Weighted Sum':<15} {'LightGBM LTR':<15} {'Delta'}")
    print("-" * 55)
    
    for metric in ndcg_weighted.keys():
        score_w = ndcg_weighted[metric]
        score_l = ndcg_ltr[metric]
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
