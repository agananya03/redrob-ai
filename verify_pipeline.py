import sys
import os
sys.path.append(os.getcwd())
import json
import warnings
warnings.filterwarnings('ignore')

from src.hybrid_ranker import HybridRanker
from src.output_writer import write_submission

print("--- Constructing High-Semantic Honeypot ---")
# A honeypot candidate with >50 years experience and a perfect match on semantics
honeypot = {
    "candidate_id": "HONEYPOT_001",
    "profile": {
        "current_title": "Senior AI Engineer",
        "years_of_experience": 55, # TRAP!
        "location": "Pune"
    },
    "skills": [
        {"name": "Python", "proficiency": "Expert", "duration_months": 24},
        {"name": "NLP", "proficiency": "Expert", "duration_months": 24}
    ],
    "career_history": [
        {
            "title": "Senior AI Engineer",
            "company": "Tech Corp",
            "start_date": "1970-01-01",
            "end_date": "2025-01-01",
            "description": "Deployed NLP models at scale."
        }
    ],
    "platform_signals": {
        "open_to_work_flag": True,
        "profile_completeness_score": 1.0,
        "recruiter_response_rate": 1.0
    }
}

jd_text = "Senior AI Engineer. 5+ years experience. Strong NLP and Python."

ranker = HybridRanker()
print("Ranking honeypot candidate...")
df = ranker.rank([honeypot], jd_text, use_ltr=False)

print("\n--- RESULTS ---")
print(f"Honeypot Final Score: {df['final_score'].iloc[0]:.4f}")
if df['final_score'].iloc[0] == 0.0:
    print("SUCCESS: Honeypot was strictly zeroed out at the end of the pipeline.")
else:
    print("FAILURE: Honeypot circumvented zeroing.")
