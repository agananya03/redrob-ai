# EXPERIMENTAL: fine-tuned on ~150 hand-labeled examples.
# Present to judges as proof-of-concept / what we'd scale with more data.
# Do NOT use the hybrid ranker's own scores as training labels —
# that would just amplify existing biases in the scorer.

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score

from src.data_loader import load_candidates, load_job_description
from src.preprocessing import build_candidate_profile
from src.evaluation import compute_ndcg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_training_pairs(labels_path, candidates_path, jd_text):
    labels_df = pd.read_csv(labels_path)
    
    if len(labels_df) < 30:
        logger.warning(f"Warning: only {len(labels_df)} labeled examples found. Fine-tuning on very little data may not improve results. Recommend labeling at least 100.")
        
    candidates = load_candidates(candidates_path)
    # Build dictionary mapping candidate_id -> profile_summary
    profile_map = {}
    for c in candidates:
        profile = build_candidate_profile(c)
        profile_map[str(profile['candidate_id'])] = profile['profile_summary']
        
    examples = []
    
    # Process labels
    for _, row in labels_df.iterrows():
        cid = str(row['candidate_id'])
        relevance = float(row['relevance'])
        
        if cid not in profile_map:
            logger.warning(f"Candidate {cid} found in labels but not in candidates. Skipping.")
            continue
            
        profile_summary = profile_map[cid]
        
        # normalize relevance to 0.0 - 1.0 (relevance is 0-3)
        normalized_label = relevance / 3.0
        
        examples.append((cid, profile_summary, normalized_label))
        
    if not examples:
        raise ValueError("No valid training examples found after matching labels to profiles.")
        
    # Split into train/eval (80/20) by candidate_id
    train_data, eval_data = train_test_split(examples, test_size=0.2, random_state=42)
    
    train_examples = []
    for cid, summary, label in train_data:
        train_examples.append(InputExample(texts=[jd_text[:2000], summary[:800]], label=label))
        
    eval_examples = []
    for cid, summary, label in eval_data:
        eval_examples.append({'candidate_id': cid, 'summary': summary, 'label': label})
        
    return train_examples, eval_examples

def finetune(train_examples, eval_examples,
             base_model='BAAI/bge-small-en-v1.5',
             output_path='data/models/finetuned_embedder',
             epochs=5,
             learning_rate=2e-5):
    
    os.makedirs('data/models', exist_ok=True)
    
    logger.info(f"Loading base model: {base_model}")
    model = SentenceTransformer(base_model)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)
    
    total_steps = len(train_dataloader) * epochs
    warmup_steps = int(total_steps * 0.1)
    
    logger.info("Starting fine-tuning...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': learning_rate},
        output_path=output_path,
        show_progress_bar=True
    )
    
    logger.info(f"Fine-tuned model saved to {output_path}")
    return SentenceTransformer(output_path)

def evaluate_before_after(eval_examples, jd_text,
                          base_model='BAAI/bge-small-en-v1.5',
                          finetuned_path='data/models/finetuned_embedder'):
                          
    logger.info("Evaluating base model...")
    base_embedder = SentenceTransformer(base_model)
    logger.info("Evaluating finetuned model...")
    ft_embedder = SentenceTransformer(finetuned_path)
    
    jd_prefix = "Represent this sentence for searching relevant passages: "
    jd_text_prefixed = jd_prefix + jd_text
    
    base_scores = []
    ft_scores = []
    labels = []
    cids = []
    
    for ex in eval_examples:
        cid = ex['candidate_id']
        summary = ex['summary']
        label = ex['label'] * 3.0 # convert back to 0-3 for evaluating with true relevance or we can just keep 0-1, NDCG only cares about order
        
        # BGE model prefix should only be on the query (jd_text) according to previous knowledge, but let's just use what's standard
        base_sum_emb = base_embedder.encode(summary, normalize_embeddings=True)
        ft_sum_emb = ft_embedder.encode(summary, normalize_embeddings=True)
        
        # JD embeddings need to be recomputed for BGE? No wait, BGE requires the prefix on the query, not the profile.
        base_jd_emb = base_embedder.encode(jd_text_prefixed, normalize_embeddings=True)
        ft_jd_emb = ft_embedder.encode(jd_text_prefixed, normalize_embeddings=True)
        
        base_score = float(np.dot(base_jd_emb, base_sum_emb.T))
        ft_score = float(np.dot(ft_jd_emb, ft_sum_emb.T))
        
        base_scores.append(base_score)
        ft_scores.append(ft_score)
        labels.append(label)
        cids.append(cid)
        
    y_true = np.asarray([labels])
    y_base = np.asarray([base_scores])
    y_ft = np.asarray([ft_scores])
    
    # K=10, but if we have fewer eval examples, actual K is min(10, len)
    actual_k = min(10, len(labels))
    base_ndcg = ndcg_score(y_true, y_base, k=actual_k) if actual_k > 1 else 0.0
    ft_ndcg = ndcg_score(y_true, y_ft, k=actual_k) if actual_k > 1 else 0.0
    
    delta = ft_ndcg - base_ndcg
    
    print("\n" + "=" * 45)
    print(f"{'Model':<26} | {'NDCG@10'}")
    print("---------------------------+--------")
    print(f"{'Base (' + base_model + ')':<26} |  {base_ndcg:.4f}")
    print(f"{'Fine-tuned (ours)':<26} |  {ft_ndcg:.4f}")
    delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
    print(f"{'Delta':<26} | {delta_str}")
    print("=" * 45 + "\n")
    
    # Plot
    os.makedirs('outputs', exist_ok=True)
    plt.figure(figsize=(8, 6))
    models = ['Base Model', 'Fine-tuned Model']
    scores = [base_ndcg, ft_ndcg]
    
    colors = ['#1f77b4' if x <= base_ndcg else '#2ca02c' for x in scores]
    if ft_ndcg < base_ndcg:
        colors[1] = '#d62728'
        
    bars = plt.bar(models, scores, color=colors)
    plt.ylabel('NDCG@10')
    plt.title('Embedding Model Comparison: Before vs After Fine-tuning')
    plt.ylim(0, 1.0)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom')
        
    plt.savefig('outputs/finetuning_ndcg_comparison.png')
    logger.info("Saved comparison chart to outputs/finetuning_ndcg_comparison.png")
    
    return {'base_ndcg': base_ndcg, 'finetuned_ndcg': ft_ndcg, 'delta': delta}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune embedding model via self-distillation")
    parser.add_argument('--labels', default='data/processed/relevance_labels.csv')
    parser.add_argument('--candidates', default='data/raw/sample_candidates.json')
    parser.add_argument('--jd', default='data/raw/job_description.docx')
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--output', default='data/models/finetuned_embedder')
    parser.add_argument('--eval_only', action='store_true', help="Skip training, just evaluate")
    args = parser.parse_args()
    
    jd_text = load_job_description(args.jd)
    train_examples, eval_examples = build_training_pairs(args.labels, args.candidates, jd_text)
    
    print(f"Training on {len(train_examples)} pairs, evaluating on {len(eval_examples)} pairs")
    
    if not args.eval_only:
        finetune(train_examples, eval_examples, output_path=args.output, epochs=args.epochs)
        
    evaluate_before_after(eval_examples, jd_text, finetuned_path=args.output)
