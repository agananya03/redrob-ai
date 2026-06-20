# Redrob Candidate Ranker

## Problem
Recruiters manually sift through thousands of profiles. Keyword filters miss the right person. We build a hybrid AI ranker.

## Approach
5-signal structured scoring (30% skill match, 20% experience, 10% education, 20% career trajectory, 20% platform signals)
+ semantic embedding similarity (all-MiniLM-L6-v2)
+ optional LLM re-rank with per-candidate reasoning (llama-3.3-70b-versatile via Groq API, free tier).
Weights are configurable. Explainable sub-scores for every candidate.

## Architecture
data_loader.py -> preprocessing.py -> structured_scoring.py
                                   -> embeddings.py
               -> hybrid_ranker.py -> output_writer.py
Optional: LLM re-rank in hybrid_ranker.py via Groq API (free).
Demo: app.py (Streamlit)

## Setup
pip install -r requirements.txt
Copy .env.example to .env and add your GROQ_API_KEY.
Place dataset files in data/raw/

## Usage
# Quick test on 50 sample candidates (no API key needed):
python run_pipeline.py --sample

# Full run on 100k candidates:
python run_pipeline.py --jd data/raw/job_description.docx

# With LLM reasoning for top 15:
python run_pipeline.py --jd data/raw/job_description.docx

# Streamlit demo:
streamlit run app.py

## Team
- Parv
