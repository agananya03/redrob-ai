import sys

new_app_content = '''import streamlit as st
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
    page_title='Candidate Ranker Pro',
    layout='wide'
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
    
    /* Streamlit Alert/Success boxes */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
        font-family: 'Inter', sans-serif !important;
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

# --- Title ---
st.markdown("<h1 class='retro-title'>✨ Candidate Ranker Pro</h1>", unsafe_allow_html=True)
st.markdown("<div class='retro-subtitle'>Enterprise-grade precision matching powered by LightGBM & Deep Semantic Embeddings</div><br>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    top_n = st.slider("Top N candidates to display", min_value=5, max_value=50, value=15)
    
    has_api_key = bool(os.getenv('GROQ_API_KEY'))
    use_llm = st.checkbox("Enable LLM Reasoning", value=has_api_key, disabled=not has_api_key)
    
    if not has_api_key:
        st.warning("GROQ_API_KEY not set in .env. LLM reasoning is disabled.")
        
    st.markdown("---")
    st.markdown("### 📂 Data Source")
    
    if 'dataset_path' not in st.session_state:
        st.session_state.dataset_path = 'data/raw/sample_candidates.json'
        
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

# --- Main Area ---

default_jd = get_jd('data/raw/job_description.docx')

# 1. Job Description Expander
with st.expander("📝 View / Edit Job Description", expanded=False):
    st.markdown("<span style='color:#64748b; font-size: 0.95rem;'>Paste or edit the job description below. The system automatically extracts core skills and experience constraints.</span>", unsafe_allow_html=True)
    jd_input = st.text_area("Job Description Details", value=default_jd, height=350, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# 2. Centered Run Button
col_spacer1, col_btn, col_spacer2 = st.columns([1, 1.5, 1])
with col_btn:
    run_btn = st.button("🚀 Analyze & Rank Candidates", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3. Results Section
results_placeholder = st.empty()

if run_btn:
    if not jd_input.strip():
        st.error("Please provide a valid job description.")
    else:
        with st.spinner("Analyzing profiles and computing scores..."):
            start_time = time.time()
            
            # Load Data
            candidates = get_candidates(st.session_state.dataset_path)
            
            # Rank
            ranker = get_ranker()
            df = ranker.rank(candidates, jd_input, top_n=top_n)
            
            # LLM Rerank
            if use_llm:
                with st.spinner("Generating AI justifications for top matches..."):
                    df = ranker.llm_rerank(df, jd_input, skip_llm=False)
            else:
                df['reasoning'] = ""
                
            elapsed = time.time() - start_time
            
            with results_placeholder.container():
                st.success(f'✅ Successfully processed {len(candidates):,} candidates in {elapsed:.2f} seconds.')
                
                # Metrics row
                st.markdown("### 📊 Ranking Overview")
                m1, m2, m3 = st.columns(3)
                top_score = df['final_score'].max() if not df.empty else 0.0
                avg_score = df['final_score'].mean() if not df.empty else 0.0
                
                m1.metric("Highest Match Score", f"{top_score:.4f}")
                m2.metric("Avg Top-N Score", f"{avg_score:.4f}")
                m3.metric("Total Profiles Analyzed", f"{len(candidates):,}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🏆 Top Recommendations")
                
                # Display DataFrame
                if not df.empty:
                    display_cols = ['rank', 'candidate_id', 'current_title', 'years_of_experience', 
                                    'final_score', 'skill_match_score', 'experience_score', 
                                    'education_score', 'trajectory_score', 'platform_signal_score', 'reasoning']
                    
                    rename_map = {
                        'rank': 'Rank',
                        'candidate_id': 'ID',
                        'current_title': 'Title',
                        'years_of_experience': 'Exp (Yrs)',
                        'final_score': 'Final Score',
                        'skill_match_score': 'Skills',
                        'experience_score': 'Exp',
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
                                            'Exp': '{:.2f}',
                                            'Edu': '{:.2f}',
                                            'Trajectory': '{:.2f}',
                                            'Platform': '{:.2f}'
                                        }),
                        use_container_width=True,
                        height=550
                    )
                    
                    # Download CSV
                    st.markdown("<br>", unsafe_allow_html=True)
                    csv_path = 'outputs/ranked_streamlit.csv'
                    write_submission(df, csv_path)
                    
                    if os.path.exists(csv_path):
                        with open(csv_path, 'rb') as f:
                            csv_bytes = f.read()
                            
                        col_dl1, col_dl2, col_dl3 = st.columns([1, 1, 1])
                        with col_dl2:
                            st.download_button(
                                label="📥 Download Results CSV",
                                data=csv_bytes,
                                file_name='ranked_candidates.csv',
                                mime='text/csv',
                                use_container_width=True
                            )
'''

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_app_content)
    
print("app.py successfully rewritten with new layout and light modern theme!")
