"""
scripts/quick_eval.py

Fast NDCG evaluation — only loads and embeds the labeled candidates
(from relevance_labels.csv), not the full 100k pool.

This runs in seconds instead of hours because it skips re-embedding
candidates you never labeled. The NDCG numbers are the same — unlabeled
candidates don't contribute to the score anyway (they're all relevance=0).

Usage:
    python scripts/quick_eval.py
    python scripts/quick_eval.py --candidates data/raw/sample_candidates.json
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_candidates, load_job_description
from src.hybrid_ranker import HybridRanker

LABELS_PATH = "data/processed/relevance_labels.csv"
K_VALUES = [10, 50, 100]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/raw/sample_candidates.json")
    parser.add_argument("--jd", default="data/raw/job_description.docx")
    parser.add_argument("--labels", default=LABELS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.labels):
        print(f"No labels found at {args.labels}. Run scripts/label_candidates.py first.")
        sys.exit(1)

    labels_df = pd.read_csv(args.labels)
    labeled_ids = set(labels_df["candidate_id"].astype(str).unique())
    print(f"Loaded {len(labels_df)} label rows across {len(labeled_ids)} unique candidates.")

    # Only load and embed the labeled candidates — skip the other 99,850
    print("Loading only labeled candidates from candidates file...")
    all_candidates = load_candidates(args.candidates)
    labeled_candidates = [
        c for c in all_candidates
        if str(c.get("candidate_id", "")) in labeled_ids
    ]
    print(f"Found {len(labeled_candidates)} labeled candidates in the candidates file.")

    if len(labeled_candidates) < 5:
        print("Too few labeled candidates found in the data file. Check candidate IDs match.")
        sys.exit(1)

    jd_text = load_job_description(args.jd)
    ranker = HybridRanker()

    print("\nRunning weighted sum (no LTR)...")
    df_weighted = ranker.rank(labeled_candidates, jd_text, use_ltr=False)

    print("Running LightGBM LTR...")
    df_ltr = ranker.rank(labeled_candidates, jd_text, use_ltr=True)

    # Build relevance map (average if labeled by multiple people)
    relevance_map = labels_df.groupby("candidate_id")["relevance"].mean().to_dict()
    relevance_map = {str(k): v for k, v in relevance_map.items()}

    def compute_ndcg(df, score_col):
        ranked_ids = df["candidate_id"].astype(str).tolist()
        true_rel = np.array([relevance_map.get(cid, 0) for cid in ranked_ids])
        pred_scores = df[score_col].fillna(0.0).values
        results = {}
        for k in K_VALUES:
            actual_k = min(k, len(df))
            if actual_k < 2:
                results[f"NDCG@{k}"] = 0.0
                continue
            results[f"NDCG@{k}"] = ndcg_score([true_rel], [pred_scores], k=actual_k)
        return results

    ndcg_w = compute_ndcg(df_weighted, "final_score")
    ndcg_ltr = compute_ndcg(df_ltr, "final_score")

    print("\n" + "=" * 60)
    print(f"{'Metric':<12} {'Weighted Sum':<16} {'LightGBM LTR':<16} {'Delta'}")
    print("-" * 60)
    for k in K_VALUES:
        key = f"NDCG@{k}"
        w = ndcg_w[key]
        l = ndcg_ltr[key]
        delta = l - w
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{key:<12} {w:<16.4f} {l:<16.4f} {delta_str}")
    print("=" * 60)

    if ndcg_ltr.get("NDCG@10", 0) > ndcg_w.get("NDCG@10", 0):
        print("\nLTR beats weighted sum — labels are working.")
    else:
        print("\nWeighted sum still ahead — labels may need more 2s and 3s to give the model signal.")


if __name__ == "__main__":
    main()