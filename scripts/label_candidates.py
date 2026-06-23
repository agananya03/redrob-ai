import os
import sys

# Add project root to sys.path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
from src.data_loader import load_candidates, load_job_description
from src.preprocessing import build_candidate_profile
from src.hybrid_ranker import HybridRanker

def main():
    parser = argparse.ArgumentParser(description="Helper script to manually label candidates")
    parser.add_argument('--labeler', required=True, help="name of the person doing the labeling")
    parser.add_argument('--data', default='data/raw/sample_candidates.json', help="path to candidates file")
    parser.add_argument('--jd', default='data/raw/job_description.docx', help="path to JD file")
    parser.add_argument('--output', default='data/processed/relevance_labels.csv', help="path to save labels")
    
    args = parser.parse_args()
    
    # 1. Run HybridRanker
    try:
        candidates = load_candidates(args.data)
        jd_text = load_job_description(args.jd)
        ranker = HybridRanker()
        df_ranked = ranker.rank(candidates, jd_text)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your pipeline runs first:\npython run_pipeline.py --sample")
        sys.exit(1)
        
    # Build skills and name map directly from candidates
    skills_map = {}
    names_map = {}
    for c in candidates:
        prof = build_candidate_profile(c)
        cid_str = str(prof['candidate_id'])
        skills_map[cid_str] = [str(name) for name in prof.get('skill_names', [])]
        names_map[cid_str] = c.get('profile', {}).get('anonymized_name', 'Unknown')
        
    # Load existing labels to avoid re-labeling
    existing_ids = set()
    if os.path.exists(args.output):
        try:
            df_existing = pd.read_csv(args.output)
            if 'candidate_id' in df_existing.columns:
                existing_ids = set(df_existing['candidate_id'].astype(str))
        except Exception:
            pass
            
    # Selection logic: top 120 + 30 random from the rest
    top_120 = df_ranked.head(120)
    outside_120 = df_ranked.iloc[120:]
    
    if not outside_120.empty:
        n_sample = min(30, len(outside_120))
        selected_outside = outside_120.sample(n=n_sample, random_state=42)
        df_combined = pd.concat([top_120, selected_outside])
    else:
        df_combined = top_120.copy()
    
    # Shuffle so labeler doesn't know the rank
    df_shuffled = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Filter out already labeled
    df_to_label = df_shuffled[~df_shuffled['candidate_id'].astype(str).isin(existing_ids)].reset_index(drop=True)
    
    pool_size = len(df_combined)
    total_to_label = len(df_to_label)
    
    if total_to_label == 0:
        print("No more candidates to label in this batch!")
        return
        
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    labeled_count = 0
    total_labeled_so_far = len(existing_ids)
    
    # Labeling loop
    for idx, row in df_to_label.iterrows():
        cid = str(row['candidate_id'])
        
        # Use standard ASCII dash to prevent Windows UnicodeEncodeError
        print('-' * 60)
        # 1-indexed count for this session vs total pool size
        print(f"Candidate {idx + 1} / {total_to_label}")
        print(f"Name:         {names_map.get(cid, 'Unknown')}")
        print(f"ID:           {cid}")
        print(f"Title:        {row.get('current_title', '')}")
        
        yoe = row.get('years_of_experience')
        yoe_str = f"{float(yoe):.1f}" if pd.notnull(yoe) else "0.0"
        print(f"Experience:   {yoe_str} years")
        
        loc = row.get('location', '')
        print(f"Location:     {loc if pd.notnull(loc) else ''}")
        
        print(f"System score: {row.get('final_score', 0):.3f}")
        print(f"Sub-scores:   skill={row.get('skill_match_score', 0):.2f}  exp={row.get('experience_score', 0):.2f}")
        print(f"              edu={row.get('education_score', 0):.2f}  traj={row.get('trajectory_score', 0):.2f}")
        print(f"              platform={row.get('platform_signal_score', 0):.2f}")
        
        top_skills = skills_map.get(cid, [])[:5]
        print(f"Top 5 skills: {', '.join(top_skills)}")
        
        summary = str(row.get('profile_summary', ''))
        print(f"Summary:      {summary.replace(chr(10), ' ')}")
        
        while True:
            choice = input("\nScore (0=no fit, 1=weak, 2=good, 3=excellent) | s=skip | q=quit+save: ").strip().lower()
            
            if choice in ['0', '1', '2', '3']:
                score = int(choice)
                
                # Append to CSV
                file_exists = os.path.exists(args.output)
                pd.DataFrame([{
                    'candidate_id': cid,
                    'relevance': score,
                    'labeled_by': args.labeler
                }]).to_csv(args.output, mode='a', header=not file_exists, index=False)
                
                labeled_count += 1
                total_labeled_so_far += 1
                break
                
            elif choice == 's':
                break
                
            elif choice == 'q':
                print(f"\nSession complete. You labeled {labeled_count} candidates this session.")
                print(f"Total labeled so far: {total_labeled_so_far} / {pool_size}")
                return
                
            else:
                print("Invalid input, try again")
                
    print(f"\nSession complete. You labeled {labeled_count} candidates this session.")
    print(f"Total labeled so far: {total_labeled_so_far} / {pool_size}")

if __name__ == "__main__":
    main()
