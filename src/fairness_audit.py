import os
import math
import pandas as pd

_detector = None

def infer_proxy_gender(name: str) -> str:
    """
    Uses gender-guesser to map the first name to a proxy gender group.
    Returns one of: 'male', 'female', 'andy', 'unknown'.
    """
    global _detector
    if _detector is None:
        import gender_guesser.detector as gender
        _detector = gender.Detector()
        
    first_name = name.strip().split()[0] if name else ""
    raw = _detector.get_gender(first_name)
    
    if raw == 'mostly_male':
        return 'male'
    elif raw == 'mostly_female':
        return 'female'
    elif raw in ['male', 'female', 'andy']:
        return raw
    return 'unknown'

def compute_exposure(ranked_df: pd.DataFrame, candidates: list[dict], top_n: int = 100) -> dict:
    """
    Computes position-weighted exposure and evaluates representation in the top N.
    """
    # 1. Build a dict: {candidate_id -> anonymized_name}
    name_map = {}
    for c in candidates:
        if 'profile' in c and 'anonymized_name' in c['profile']:
            name_map[c['candidate_id']] = c['profile']['anonymized_name']
        else:
            name_map[c['candidate_id']] = "Unknown"
            
    # Pre-compute full pool counts to calculate percentage later
    full_pool_counts = {'male': 0, 'female': 0, 'andy': 0, 'unknown': 0}
    for c in candidates:
        name = name_map.get(c['candidate_id'], "Unknown")
        gen = infer_proxy_gender(name)
        full_pool_counts[gen] += 1
        
    total_pool = sum(full_pool_counts.values())
    
    # 2. Extract Top N and compute exposure
    top_df = ranked_df.head(top_n)
    
    top_n_counts = {'male': 0, 'female': 0, 'andy': 0, 'unknown': 0}
    exposures = {'male': [], 'female': [], 'andy': [], 'unknown': []}
    
    for i, row in top_df.iterrows():
        rank = row.get('rank', i + 1)
        cid = row['candidate_id']
        name = name_map.get(cid, "Unknown")
        gen = infer_proxy_gender(name)
        
        # 3. Position-weighted exposure: 1 / log2(1 + rank)
        exposure = 1.0 / math.log2(1 + rank)
        
        top_n_counts[gen] += 1
        exposures[gen].append(exposure)
        
    # 4 & 5. Print a clear summary table
    print(f"\n{'Group':<10} | {'Pool Count':<10} | {'Top-N Count':<11} | {'% of Top-N':<10} | {'Avg Exposure'}")
    print("-" * 65)
    
    summary = {}
    for g in ['male', 'female', 'andy', 'unknown']:
        p_count = full_pool_counts[g]
        t_count = top_n_counts[g]
        
        pct_top = (t_count / top_n) * 100 if top_n > 0 else 0
        pct_pool = (p_count / total_pool) * 100 if total_pool > 0 else 0
        
        avg_exp = sum(exposures[g]) / len(exposures[g]) if len(exposures[g]) > 0 else 0.0
        
        print(f"{g:<10} | {p_count:<10} | {t_count:<11} | {pct_top:>5.1f}%     | {avg_exp:.4f}")
        
        summary[g] = {
            'pool_count': p_count,
            'top_n_count': t_count,
            'pct_of_top_n': pct_top,
            'avg_exposure': avg_exp
        }
        
        # 6. Flag significant deviations (>15 percentage points)
        if abs(pct_top - pct_pool) > 15:
            over_under = "over" if pct_top > pct_pool else "under"
            print(f"  WARNING: {g} is {over_under}-represented in top {top_n}")
            
    return summary

def verify_name_not_in_scoring(codebase_path: str = 'src') -> bool:
    """
    Scans src/ to ensure 'anonymized_name' is not being used to score candidates.
    """
    clean = True
    for root, dirs, files in os.walk(codebase_path):
        for f in files:
            if f.endswith('.py') and f != 'fairness_audit.py':
                filepath = os.path.join(root, f)
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if 'anonymized_name' in content:
                        print(f"WARNING: anonymized_name found in {os.path.basename(filepath)} — check this!")
                        clean = False
                        
    if clean:
        print("OK: anonymized_name is not used in any scoring file.")
        
    return clean

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import load_candidates, load_job_description
    from src.hybrid_ranker import HybridRanker
    
    candidates = load_candidates('data/raw/sample_candidates.json')
    jd_text = load_job_description('data/raw/job_description.docx')
    
    ranker = HybridRanker()
    ranked_df = ranker.rank(candidates, jd_text, use_ltr=True)
    
    verify_name_not_in_scoring()
    compute_exposure(ranked_df, candidates, top_n=50)
