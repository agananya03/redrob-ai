import streamlit as st
import os
import time
import pandas as pd
from dotenv import load_dotenv

# Load dependencies securely with caching
from src.hybrid_ranker import HybridRanker
from src.data_loader import load_candidates, load_job_description
from src.output_writer import write_submission
import io

load_dotenv()

st.set_page_config(
    page_title='Redrob Candidate Ranker',
    layout='wide'
)

# --- Caching ---
@st.cache_resource
def get_ranker():
    return HybridRanker()

@st.cache_data
def get_candidates(file_path):
    return load_candidates(file_path)

@st.cache_data
def get_jd(file_path):
    try:
        return load_job_description(file_path)
    except Exception:
        return ""

# --- Title ---
st.title("Redrob AI — Candidate Ranking Demo")
st.markdown("### Hybrid scoring: structured signals + semantic embeddings")

# --- Sidebar ---
st.sidebar.header("Controls")
top_n = st.sidebar.slider("Top N candidates to show", min_value=5, max_value=50, value=15)

has_api_key = bool(os.getenv('GROQ_API_KEY'))
use_llm = st.sidebar.checkbox("Use LLM reasoning", value=False, disabled=not has_api_key)

if not has_api_key:
    st.sidebar.warning("GROQ_API_KEY not set in .env. LLM reasoning is disabled.")

if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = 'data/raw/sample_candidates.json'

if st.sidebar.button("Load sample data (50 candidates)"):
    st.session_state.dataset_path = 'data/raw/sample_candidates.json'
    st.sidebar.success("Sample data selected!")

if st.sidebar.button("Load full dataset (100k candidates)"):
    st.session_state.dataset_path = 'data/raw/candidates.jsonl'
    st.sidebar.warning("Full dataset selected! This will be slow.")

# --- Main Area ---
col1, col2 = st.columns([0.4, 0.6])

with col1:
    st.subheader("Job Description")
    default_jd = get_jd('data/raw/job_description.docx')
    jd_input = st.text_area("Paste job description here...", value=default_jd, height=400)

with col2:
    st.subheader("Results")
    results_placeholder = st.empty()

# --- Run Ranking ---
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([0.4, 0.2, 0.4])
with col_btn2:
    run_btn = st.button("Run Ranking", use_container_width=True)

if run_btn:
    if not jd_input.strip():
        st.error("Please enter a job description.")
    else:
        with st.spinner("Ranking candidates..."):
            start_time = time.time()
            
            # 1. Load Data
            candidates = get_candidates(st.session_state.dataset_path)
            
            # 2. Rank
            ranker = get_ranker()
            df = ranker.rank(candidates, jd_input, top_n=top_n)
            
            # 3. LLM Rerank
            if use_llm:
                df = ranker.llm_rerank(df, jd_input, skip_llm=False)
            else:
                df['reasoning'] = ""
                
            elapsed = time.time() - start_time
            
            with results_placeholder.container():
                st.success(f'Ranked {len(candidates)} candidates in {elapsed:.1f}s')
                
                # Metric row
                m1, m2, m3 = st.columns(3)
                top_score = df['final_score'].max() if not df.empty else 0.0
                avg_score = df['final_score'].mean() if not df.empty else 0.0
                m1.metric("Top score", f"{top_score:.4f}")
                m2.metric("Avg score", f"{avg_score:.4f}")
                m3.metric("Candidates evaluated", len(candidates))
                
                # Display DataFrame
                if not df.empty:
                    display_cols = ['rank', 'candidate_id', 'current_title', 'years_of_experience', 
                                    'final_score', 'skill_match_score', 'experience_score', 
                                    'education_score', 'trajectory_score', 'platform_signal_score', 'reasoning']
                    
                    rename_map = {
                        'rank': 'Rank',
                        'candidate_id': 'Candidate ID',
                        'current_title': 'Title',
                        'years_of_experience': 'Exp (yrs)',
                        'final_score': 'Final Score',
                        'skill_match_score': 'Skill',
                        'experience_score': 'Exp',
                        'education_score': 'Education',
                        'trajectory_score': 'Trajectory',
                        'platform_signal_score': 'Platform',
                        'reasoning': 'Reasoning'
                    }
                    
                    # Ensure columns exist to prevent errors if ranking output changes
                    missing_cols = [c for c in display_cols if c not in df.columns]
                    for mc in missing_cols:
                        df[mc] = 0.0 if mc != 'reasoning' else ''
                        
                    df_display = df[display_cols].rename(columns=rename_map)
                    
                    st.dataframe(
                        df_display.style.background_gradient(subset=['Final Score'], cmap='viridis')
                                        .format({
                                            'Final Score': '{:.2f}',
                                            'Skill': '{:.2f}',
                                            'Exp': '{:.2f}',
                                            'Education': '{:.2f}',
                                            'Trajectory': '{:.2f}',
                                            'Platform': '{:.2f}'
                                        }),
                        use_container_width=True
                    )
                    
                    # Download CSV using write_submission logic but generating bytes
                    csv_path = 'outputs/ranked_streamlit.csv'
                    write_submission(df, csv_path)
                    
                    if os.path.exists(csv_path):
                        with open(csv_path, 'rb') as f:
                            csv_bytes = f.read()
                            
                        st.download_button(
                            label="Download Results CSV",
                            data=csv_bytes,
                            file_name='ranked_candidates.csv',
                            mime='text/csv'
                        )
