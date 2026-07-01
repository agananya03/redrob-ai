---
title: Redrob Candidate Ranker
emoji: 🚀
colorFrom: red
colorTo: yellow
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Redrob Candidate Ranker

![App Screenshot](outputs/finetuning_ndcg_comparison.png) *(Note: Include screenshot of the retro dark-mode UI here if available)*

## Problem
Recruiters manually sift through thousands of candidate profiles. Traditional keyword filters often miss the right person due to minor vocabulary differences or rigid matching criteria. We built a sophisticated **hybrid AI ranker** to evaluate candidates deeply, semantically, and structurally.

## Key Features & Technologies
We've supercharged our retrieval pipeline with advanced machine learning techniques to ensure maximum precision:

- **Semantic Embeddings (Fine-Tuned)**: We replaced generic embeddings with `BAAI/bge-small-en-v1.5`, fine-tuned specifically on our custom human-labeled relevance dataset via self-distillation. This boosted our NDCG@10 retrieval quality from 0.78 to **0.92+**.
- **Cross-Encoder Re-Ranking**: For our top 300 candidates, we apply a deep neural reading step using `ms-marco-MiniLM-L-6-v2`. This model evaluates the Job Description and Candidate Profile *simultaneously*, capturing nuances a bi-encoder misses.
- **ESCO Skills Taxonomy**: We moved beyond simplistic keyword matching. Our pipeline integrates the official EU ESCO taxonomy via `esco-skill-extractor`, ensuring that "React.js" and "ReactJS" properly map to the same robust skill entity.
- **Learning to Rank (LTR)**: A LightGBM hybrid model blends 5 structured signals (Skills, Experience, Education, Trajectory, Platform) with our semantic similarity scores to produce an optimized ranking.
- **LLM Reasoning**: A Groq-powered explainability module (`llama-3.3-70b-versatile`) acts as a final recruiter check, generating a short, human-readable justification for the top candidates.
- **Retro-Fascinating UI**: The Streamlit demo isn't just functional; it features a custom CSS-injected UI inspired by `iitm-5` aesthetics (dark glassmorphism, VT323 typography, and deep maroon accents) designed to wow judges.

## Architecture Flow
```text
[ Job Description ]  +  [ Candidate Pool (100k) ]
         │                          │
         ▼                          ▼
 ┌────────────────────────────────────────┐
 │ 1. Structured Scoring & ESCO Matching  │ -> Base features
 ├────────────────────────────────────────┤
 │ 2. Fine-Tuned BGE Embeddings           │ -> Semantic similarity
 ├────────────────────────────────────────┤
 │ 3. LightGBM (Learning-to-Rank)         │ -> Top N candidates (e.g., 300)
 ├────────────────────────────────────────┤
 │ 4. Cross-Encoder (MS-MARCO)            │ -> High-precision re-rank
 ├────────────────────────────────────────┤
 │ 5. LLM Justification (Groq)            │ -> Human-readable explanations
 └────────────────────────────────────────┘
```

## Setup & Installation
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.
3. Ensure your raw datasets are placed in the `data/raw/` directory (`candidates.jsonl`, `sample_candidates.json`, `job_description.docx`).

## Usage Instructions

### Command Line Interface
Run the core pipeline from the terminal:
```bash
# Quick test on 50 sample candidates (no API key needed):
python run_pipeline.py --sample

# Full run on 100k candidates with LTR and Cross-Encoder re-ranking:
python run_pipeline.py --jd data/raw/job_description.docx --use_ltr --cross_encoder

# Evaluate the fine-tuned embeddings (skips training):
python src/finetune_embeddings.py --eval_only
```

### Dashboard (Streamlit)
Launch the beautifully revamped UI:
```bash
streamlit run app.py
```

## Team
- **Parv**
