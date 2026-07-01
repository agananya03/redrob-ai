import sys

new_app_content = '''import streamlit as st
import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load dependencies securely with caching
from src.hybrid_ranker import HybridRanker
from src.data_loader import load_candidates, load_job_description
from src.output_writer import write_submission
from src.fairness_audit import compute_exposure
from src.ltr_ranker import load_model, FEATURE_NAMES

load_dotenv()

st.set_page_config(
    page_title='Candidate Ranker Pro',
    layout='wide',
    initial_sidebar_state='expanded'
)

def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #f4f7f6;
        color: #2b2d42;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Typography */
    .retro-title {
        font-family: 'Inter', sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1a1a24 !important;
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }
    
    .retro-subtitle {
        font-family: 'Inter', sans-serif !important;
        color: #4a4e69 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-top: 0px !important;
    }
    
    /* Subheaders */
    h1, h2, h3, .st-emotion-cache-1629p8f h2 {
        font-family: 'Inter', sans-serif !important;
        color: #2b2d42 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0;
        box-shadow: 2px 0 15px rgba(0,0,0,0.03);
    }
    [data-testid="stSidebar"] * {
        font-family: 'Inter', sans-serif !important;
        color: #2b2d42 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        color: #2b2d42 !important;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.02);
    }
    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] > div:nth-child(2) {
        border: 1px solid #e2e8f0;
        border-top: none;
        border-radius: 0 0 8px 8px;
        background: #ffffff;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    
    /* Text Inputs */
    .stTextArea textarea {
        background-color: #f8fafc !important;
        color: #1e293b !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        padding: 15px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2) !important;
        background-color: #ffffff !important;
    }
    
    /* Primary Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.25) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
    }
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif !important;
        color: #1e293b !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem !important;
    }
    
    /* DataFrame */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Hide top header line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

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

# Initialize Session State
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = 'data/raw/sample_candidates.json'
if 'ranked_df' not in st.session_state:
    st.session_state.ranked_df = None
if 'raw_candidates' not in st.session_state:
    st.session_state.raw_candidates = None
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = get_jd('data/raw/job_description.docx')

# --- PAGES ---

def page_ranker():
    st.markdown("<h1 class='retro-title'>✨ Candidate Ranker Pro</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Enterprise-grade precision matching powered by LightGBM & Deep Semantic Embeddings</div><br>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        top_n = st.slider("Top N candidates to display", min_value=5, max_value=50, value=15)
        
        has_api_key = bool(os.getenv('GROQ_API_KEY'))
        use_llm = st.checkbox("Enable LLM Reasoning", value=has_api_key, disabled=not has_api_key)
        
        if not has_api_key:
            st.warning("GROQ_API_KEY not set in .env. LLM reasoning is disabled.")
            
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        
        dataset_choice = st.radio(
            "Select Candidates Dataset:",
            ("Sample Data (50 rows)", "Full Data (100k rows)"),
            index=0 if 'sample' in st.session_state.dataset_path else 1
        )
        
        if dataset_choice == "Sample Data (50 rows)":
            st.session_state.dataset_path = 'data/raw/sample_candidates.json'
        else:
            st.session_state.dataset_path = 'data/raw/candidates.jsonl'
            st.warning("⚠️ Full dataset selected. Ranking may take longer.")

    # Job Description Expander
    with st.expander("📝 View / Edit Job Description", expanded=False):
        st.markdown("<span style='color:#64748b; font-size: 0.95rem;'>Paste or edit the job description below. The system automatically extracts core skills and experience constraints.</span>", unsafe_allow_html=True)
        jd_input = st.text_area("Job Description Details", value=st.session_state.jd_text, height=350, label_visibility="collapsed")
        st.session_state.jd_text = jd_input

    st.markdown("<br>", unsafe_allow_html=True)

    col_spacer1, col_btn, col_spacer2 = st.columns([1, 1.5, 1])
    with col_btn:
        run_btn = st.button("🚀 Analyze & Rank Candidates", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    results_placeholder = st.empty()

    if run_btn:
        if not jd_input.strip():
            st.error("Please provide a valid job description.")
        else:
            with st.spinner("Analyzing profiles and computing scores..."):
                start_time = time.time()
                candidates = get_candidates(st.session_state.dataset_path)
                st.session_state.raw_candidates = candidates
                ranker = get_ranker()
                
                df = ranker.rank(candidates, jd_input, top_n=top_n)
                
                if use_llm:
                    with st.spinner("Generating AI justifications for top matches..."):
                        df = ranker.llm_rerank(df, jd_input, skip_llm=False)
                else:
                    df['reasoning'] = ""
                    
                st.session_state.ranked_df = df
                elapsed = time.time() - start_time
                
                with results_placeholder.container():
                    st.success(f'✅ Successfully processed {len(candidates):,} candidates in {elapsed:.2f} seconds.')
                    
    # Render results if we have them
    if st.session_state.ranked_df is not None:
        df = st.session_state.ranked_df
        st.markdown("### 📊 Ranking Overview")
        m1, m2, m3 = st.columns(3)
        top_score = df['final_score'].max() if not df.empty else 0.0
        avg_score = df['final_score'].mean() if not df.empty else 0.0
        
        m1.metric("Highest Match Score", f"{top_score:.4f}")
        m2.metric("Avg Top-N Score", f"{avg_score:.4f}")
        m3.metric("Total Profiles Analyzed", f"{len(st.session_state.raw_candidates) if st.session_state.raw_candidates else len(df):,}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🏆 Top Recommendations")
        
        display_cols = ['rank', 'candidate_id', 'current_title', 'years_of_experience', 
                        'final_score', 'skill_match_score', 'experience_score', 
                        'education_score', 'trajectory_score', 'platform_signal_score', 'reasoning']
        
        rename_map = {
            'rank': 'Rank',
            'candidate_id': 'ID',
            'current_title': 'Title',
            'years_of_experience': 'Exp',
            'final_score': 'Final Score',
            'skill_match_score': 'Skills',
            'experience_score': 'Exp Score',
            'education_score': 'Edu',
            'trajectory_score': 'Trajectory',
            'platform_signal_score': 'Platform',
            'reasoning': 'AI Reasoning'
        }
        
        missing_cols = [c for c in display_cols if c not in df.columns]
        for mc in missing_cols:
            df[mc] = 0.0 if mc != 'reasoning' else ''
            
        df_display = df[display_cols].rename(columns=rename_map)
        
        st.dataframe(
            df_display.style.background_gradient(subset=['Final Score'], cmap='Blues')
                            .format({
                                'Final Score': '{:.2f}',
                                'Skills': '{:.2f}',
                                'Exp Score': '{:.2f}',
                                'Edu': '{:.2f}',
                                'Trajectory': '{:.2f}',
                                'Platform': '{:.2f}'
                            }),
            use_container_width=True,
            height=500
        )

def page_fairness():
    st.markdown("<h1 class='retro-title'>⚖️ Fairness & Bias Audit</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Automated structural demographic analysis on ranking outcomes.</div><br>", unsafe_allow_html=True)
    
    if st.session_state.ranked_df is None or st.session_state.raw_candidates is None:
        st.warning("⚠️ Please run the Ranker Pipeline first to generate data for the audit.")
        return
        
    df = st.session_state.ranked_df
    raw_candidates = st.session_state.raw_candidates
    
    st.markdown("### 🛡️ Demographic Exposure Report")
    st.write("This module utilizes gender-guesser as a proxy to analyze if the LightGBM model is heavily skewing visibility toward any specific demographic group in the Top N.")
    
    try:
        from src.fairness_audit import compute_exposure
        with st.spinner("Computing exposure metrics..."):
            exposure_stats = compute_exposure(df, raw_candidates, top_n=min(50, len(df)))
            
        st.markdown(f"**Baseline Pool Evaluated:** {exposure_stats['total_pool_size']:,} candidates")
        st.markdown(f"**Top N Window Analyzed:** {exposure_stats['top_n']}")
        
        # Build DataFrame for charts
        metrics_list = []
        for gender, exp_val in exposure_stats['exposure'].items():
            baseline_pct = exposure_stats['baseline_distribution'].get(gender, 0.0)
            metrics_list.append({
                'Demographic': gender.capitalize(),
                'Top N Exposure': exp_val,
                'Baseline %': baseline_pct
            })
            
        metric_df = pd.DataFrame(metrics_list)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Position-Weighted Exposure (Top N)")
            st.bar_chart(metric_df.set_index('Demographic')['Top N Exposure'], color="#2563eb")
        with c2:
            st.markdown("#### Baseline Distribution (Full Pool)")
            st.bar_chart(metric_df.set_index('Demographic')['Baseline %'], color="#64748b")
            
        st.info("💡 **Interpretation:** If the Exposure heavily outpaces the Baseline % for a specific group, the model may be exhibiting systemic bias that requires score calibration.")
    except Exception as e:
        st.error(f"Failed to run fairness audit: {e}")

def page_analytics():
    st.markdown("<h1 class='retro-title'>📊 Model Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Deep dive into LightGBM feature importance and system ablation.</div><br>", unsafe_allow_html=True)
    
    st.markdown("### 🧠 LightGBM Feature Importance")
    try:
        model = load_model('data/processed/ltr_model.pkl')
        importances = model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Gain Importance': importances})
        imp_df = imp_df.sort_values(by='Gain Importance', ascending=False)
        
        st.bar_chart(imp_df.set_index('Feature')['Gain Importance'], color="#3b82f6")
        
        st.write("This chart represents the raw information gain each structured/semantic signal provided to the Learning-to-Rank trees during training.")
    except Exception as e:
        st.error(f"Could not load LTR model for feature importance: {e}")
        
    st.markdown("---")
    st.markdown("### 🔬 Pipeline Ablation Impact")
    st.write("Based on our latest evaluation (quick_eval.py), removing specific components impacts the NDCG@10 retrieval quality.")
    
    ablation_data = pd.DataFrame({
        'Component': ['Baseline (BGE-Small)', 'Fine-tuned Semantic Embeddings', 'Full LightGBM LTR'],
        'NDCG@10': [0.7816, 0.9233, 0.8689]
    }).set_index('Component')
    
    st.bar_chart(ablation_data, color="#10b981")

def page_deep_dive():
    st.markdown("<h1 class='retro-title'>🔍 Candidate Deep Dive</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Detailed structural breakdown of individual candidate scores.</div><br>", unsafe_allow_html=True)
    
    if st.session_state.ranked_df is None:
        st.warning("⚠️ Please run the Ranker Pipeline first.")
        return
        
    df = st.session_state.ranked_df
    candidate_ids = df['candidate_id'].tolist()
    
    selected_cid = st.selectbox("Select Candidate ID to analyze:", candidate_ids)
    
    if selected_cid:
        row = df[df['candidate_id'] == selected_cid].iloc[0]
        st.markdown(f"### Profile: **{row['current_title']}** ({selected_cid})")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Final Calibrated Score", f"{row['final_score']:.3f}")
        c2.metric("Overall Rank", f"#{row['rank']}")
        c3.metric("Total Experience", f"{row['years_of_experience']} Yrs")
        
        st.markdown("#### 🎯 Score Breakdown")
        
        score_df = pd.DataFrame({
            'Category': ['Skills', 'Experience', 'Education', 'Trajectory', 'Platform', 'Semantic (Context)'],
            'Score': [
                row['skill_match_score'], row['experience_score'], 
                row['education_score'], row['trajectory_score'], 
                row['platform_signal_score'], row['semantic_score']
            ]
        })
        
        # Display as a bar chart to simulate a profile radar
        st.bar_chart(score_df.set_index('Category')['Score'], color="#8b5cf6")
        
        st.markdown("#### 💬 LLM Recruiter Reasoning")
        st.info(row['reasoning'] if row['reasoning'] else "LLM Reasoning was disabled during ranking.")
        
        st.markdown("#### 📄 Profile Summary Snippet")
        st.text_area("Summary", row.get('profile_summary', 'N/A'), height=200, disabled=True)

# --- ROUTER ---
pages = {
    "🚀 Ranker Pipeline": page_ranker,
    "🔍 Candidate Deep Dive": page_deep_dive,
    "⚖️ Fairness Audit": page_fairness,
    "📊 Model Analytics": page_analytics
}

with st.sidebar:
    st.markdown("### 🧭 Navigation")
    selection = st.radio("Go to:", list(pages.keys()), label_visibility="collapsed")

pages[selection]()
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_app_content)

print("Multipage app generated!")
