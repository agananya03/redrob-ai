import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_loader import load_candidates, load_job_description
from src.hybrid_ranker import HybridRanker
from src.evaluation import compute_ndcg

def run_ablation_study():
    jd_path = 'data/raw/job_description.docx'
    candidates_path = 'data/raw/sample_candidates.json'
    output_path = 'outputs/ablation_results.csv'
    
    candidates = load_candidates(candidates_path)
    jd_text = load_job_description(jd_path)
    
    print(f"Loaded {len(candidates)} candidates. Running ablation study...")
    
    # Configurations to test
    configs = [
        {'name': 'Baseline (Full LTR)', 'use_ltr': True, 'ablation_feature': None},
        {'name': '- Platform Signal', 'use_ltr': True, 'ablation_feature': 'platform_signal_score'},
        {'name': '- Semantic Score', 'use_ltr': True, 'ablation_feature': 'semantic_score'},
        {'name': '- Trajectory Score', 'use_ltr': True, 'ablation_feature': 'trajectory_score'},
        {'name': '- LightGBM (Weighted Sum)', 'use_ltr': False, 'ablation_feature': None},
    ]
    
    results = []
    baseline_ndcg = None
    
    ranker = HybridRanker()
    
    for conf in configs:
        print(f"\nRunning configuration: {conf['name']}")
        df = ranker.rank(
            candidates, 
            jd_text, 
            use_ltr=conf['use_ltr'], 
            ablation_feature=conf['ablation_feature']
        )
        
        ndcg_res = compute_ndcg(df, k_values=[10])
        score = ndcg_res.get('ndcg@10', 0.0)
        
        if conf['name'] == 'Baseline (Full LTR)':
            baseline_ndcg = score
            delta = 0.0
        else:
            delta = score - baseline_ndcg
            
        results.append({
            'Configuration': conf['name'],
            'NDCG@10': score,
            'Delta': delta
        })
        
    # Print clear summary table
    print("\n" + "=" * 60)
    print(f"{'Configuration':<30} | {'NDCG@10':<10} | {'Delta':<10}")
    print("-" * 60)
    for r in results:
        delta_str = f"+{r['Delta']:.4f}" if r['Delta'] > 0 else f"{r['Delta']:.4f}"
        if r['Configuration'] == 'Baseline (Full LTR)':
            delta_str = "    -"
        print(f"{r['Configuration']:<30} | {r['NDCG@10']:<10.4f} | {delta_str}")
    print("=" * 60)
    
    # Save to CSV
    os.makedirs('outputs', exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    run_ablation_study()
