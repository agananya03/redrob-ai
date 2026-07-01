import argparse
import time
from src.data_loader import load_job_description, load_candidates
from src.hybrid_ranker import HybridRanker
from src.output_writer import write_submission
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_on_sample(use_ltr=None, use_cross_encoder=False):
    """
    Convenience function that runs the full pipeline on sample_candidates.json
    with skip_llm=True.
    """
    start_time = time.time()
    jd_path = 'data/raw/job_description.docx'
    candidates_path = 'data/raw/sample_candidates.json'
    output_path = 'outputs/ranked_sample_candidates.csv'
    
    jd_text = load_job_description(jd_path)
    candidates = load_candidates(candidates_path)
    
    print(f"Loaded {len(candidates)} candidates. Running hybrid scoring...")
    
    ranker = HybridRanker()
    ranked = ranker.rank(candidates, jd_text, top_n=100, use_ltr=use_ltr, use_cross_encoder=use_cross_encoder)
    
    # skip_llm=True
    ranked = ranker.llm_rerank(ranked, jd_text, skip_llm=True)
    
    write_submission(ranked, output_path)
    
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

def run():
    parser = argparse.ArgumentParser(description="Candidate Ranking Pipeline")
    parser.add_argument('--jd', type=str, default='data/raw/job_description.docx',
                        help='Path to the job description docx')
    parser.add_argument('--candidates', type=str, default='data/raw/candidates.jsonl',
                        help='Path to the candidates data')
    parser.add_argument('--top_n', type=int, default=100,
                        help='Number of top candidates to output')
    parser.add_argument('--skip_llm', action='store_true',
                        help='Skip LLM re-ranking')
    parser.add_argument('--output', type=str, default='outputs/ranked_candidates.csv',
                        help='Path to the output CSV')
    parser.add_argument('--sample', action='store_true',
                        help='Run on sample_candidates.json with skip_llm=True')
    parser.add_argument('--use_ltr', dest='use_ltr', action='store_true', default=None,
                        help='Force the LightGBM LTR ranker (Checkpoint B). Fails over to '
                             'the weighted sum if relevance_labels.csv is missing/too small.')
    parser.add_argument('--no_ltr', dest='use_ltr', action='store_false',
                        help='Force the hand-set weighted sum, even if relevance_labels.csv exists.')
    parser.add_argument('--cross_encoder', action='store_true',
                        help='Use CrossEncoder to rerank the top 300 candidates.')

    args = parser.parse_args()

    if args.sample:
        run_on_sample(use_ltr=args.use_ltr, use_cross_encoder=args.cross_encoder)
        return

    start_time = time.time()
    
    jd_text = load_job_description(args.jd)
    candidates = load_candidates(args.candidates)
    
    print(f"Loaded {len(candidates)} candidates. Running hybrid scoring...")
    
    ranker = HybridRanker()
    ranked = ranker.rank(candidates, jd_text, top_n=args.top_n, use_ltr=args.use_ltr, use_cross_encoder=args.cross_encoder)
    
    if not args.skip_llm:
        ranked = ranker.llm_rerank(ranked, jd_text)
    else:
        ranked = ranker.llm_rerank(ranked, jd_text, skip_llm=True)
        
    write_submission(ranked, args.output)
    
    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    load_dotenv()
    run()
    