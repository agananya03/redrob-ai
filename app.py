import streamlit as st
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

# Initialize Session State
if 'dataset_path' not in st.session_state:
    st.session_state.dataset_path = 'data/raw/candidates.jsonl'
if 'ranked_df' not in st.session_state:
    st.session_state.ranked_df = None
if 'raw_candidates' not in st.session_state:
    st.session_state.raw_candidates = None
if 'jd_text' not in st.session_state:
    st.session_state.jd_text = "Data Scientist with Python and SQL"
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

# CSS INJECTION
def inject_custom_css():
    theme = st.session_state.theme
    
    if theme == 'light':
        bg_primary = '#ffffff'
        bg_sidebar = '#f8f4ed'
        text_primary = '#1a1a1a'
        text_muted = '#6c757d'
        accent = '#e56b40'
        accent_hover = '#f95955'
        border = 'rgba(0, 0, 0, 0.06)'
        card_bg = '#ffffff'
        shadow = '0 1px 3px rgb(0 0 0 / 0.04), 0 4px 14px rgb(0 0 0 / 0.05)'
        particle_color = '#e56b40'
    else:
        bg_primary = '#1E1B1E'
        bg_sidebar = '#191619'
        text_primary = '#E5E1D8'
        text_muted = '#9E9A91'
        accent = '#e56b40'  # Keep the orange accent for pop
        accent_hover = '#c44040'
        border = '#3A343A'
        card_bg = '#2A262A'
        shadow = '0 4px 8px rgba(0, 0, 0, 0.3)'
        particle_color = '#E5E1D8'

    css = f"""
    <style>
    @import url('https://fonts.cdnfonts.com/css/chillax');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    body {{
        background-color: {bg_primary} !important;
    }}
    
    .stApp {{
        background-color: {bg_primary} !important;
        font-family: 'DM Sans', sans-serif !important;
    }}
    
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp table, .stApp th, .stApp td, .stApp tr, .stApp b {{
        color: {text_primary} !important;
    }}
    
    .retro-title {{
        font-family: 'Chillax', system-ui, sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }}
    
    .retro-subtitle {{
        font-family: 'DM Sans', sans-serif !important;
        color: {accent} !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-top: 0px !important;
    }}
    
    h1, h2, h3, .st-emotion-cache-1629p8f h2 {{
        font-family: 'Chillax', system-ui, sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {bg_sidebar} !important;
        border-right: 1px solid {border};
    }}
    
    .streamlit-expanderHeader {{
        background-color: {bg_primary} !important;
        border-radius: 1.125rem;
        border: 1px solid {border} !important;
    }}
    
    div[data-testid="stExpander"] > div:nth-child(2) {{
        border: 1px solid {border} !important;
        border-top: none !important;
        border-radius: 0 0 1.125rem 1.125rem;
        background: {bg_primary} !important;
        padding: 1.25rem;
    }}
    
    .stTextArea textarea {{
        background-color: {bg_primary} !important;
        color: {text_primary} !important;
        border: 1px solid {border} !important;
        border-radius: 0.625rem !important;
        padding: 1rem !important;
    }}
    .stTextArea textarea:disabled {{
        background-color: {card_bg} !important;
        color: {text_primary} !important;
        opacity: 1 !important;
        -webkit-text-fill-color: {text_primary} !important;
    }}
    .stTextArea textarea:focus {{
        border-color: {accent} !important;
    }}
    
    div.stButton > button, div.stDownloadButton > button {{
        background: {accent} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.625rem !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: {shadow} !important;
    }}
    div.stButton > button:hover, div.stDownloadButton > button:hover {{
        background: {accent_hover} !important;
        transform: translateY(-2px);
    }}
    div.stButton > button *, div.stDownloadButton > button * {{
        color: #ffffff !important;
    }}
    
    /* Dynamic Metric Containers */
    div[data-testid="metric-container"] {{
        background-color: {card_bg};
        border: 1px solid {border};
        padding: 1.25rem;
        border-radius: 1.125rem;
        box-shadow: {shadow};
    }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] * {{
        font-family: 'Chillax', system-ui, sans-serif !important;
        color: {accent} !important;
        font-size: 2.2rem !important;
    }}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] * {{
        color: {text_muted} !important;
        text-transform: uppercase;
        font-size: 0.85rem !important;
    }}
    
    /* Custom Candidate Cards */
    .candidate-card {{
        background-color: {card_bg};
        border: 1px solid {border};
        border-radius: 1.125rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: {shadow};
        transition: transform 0.2s ease;
    }}
    .candidate-card:hover {{
        transform: translateY(-3px);
    }}
    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        border-bottom: 1px solid {border};
        padding-bottom: 0.5rem;
    }}
    .card-title {{
        font-family: 'Chillax', system-ui, sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: {text_primary};
        margin: 0;
    }}
    .card-score {{
        font-family: 'Chillax', system-ui, sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        color: {accent};
        margin: 0;
    }}
    .score-grid {{
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 10px;
        margin-bottom: 1rem;
    }}
    .score-box {{
        background: {bg_primary};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }}
    .score-label {{
        font-size: 0.75rem;
        color: {text_muted};
        text-transform: uppercase;
        margin-bottom: 4px;
        display: block;
    }}
    .score-value {{
        font-size: 1.2rem;
        font-weight: 600;
        color: {text_primary};
    }}
    .card-reasoning {{
        background: {bg_primary};
        border-left: 4px solid {accent};
        padding: 1rem;
        border-radius: 4px;
        font-size: 0.95rem;
        color: {text_primary};
        font-style: italic;
    }}
    
    details.candidate-row {{
        background-color: {card_bg} !important;
        border: 1px solid {border} !important;
        border-radius: 8px;
        margin-bottom: 8px;
        overflow: hidden;
        color: {text_primary} !important;
    }}
    details.candidate-row summary {{
        display: grid;
        grid-template-columns: 80px 3fr 1fr;
        padding: 1rem 1.5rem;
        cursor: pointer;
        background-color: {bg_primary} !important;
        font-weight: 600;
        align-items: center;
        list-style: none;
    }}
    details.candidate-row summary::-webkit-details-marker {{
        display: none;
    }}
    details.candidate-row summary:hover {{
        background-color: {card_bg} !important;
    }}
    .summary-rank {{
        font-size: 1.1rem;
        color: {accent} !important;
    }}
    .summary-name {{
        font-size: 1.05rem;
        color: {text_primary} !important;
    }}
    .summary-score {{
        text-align: right;
        font-size: 1.1rem;
        color: {text_primary} !important;
    }}
    .details-content {{
        padding: 1.5rem;
        border-top: 1px solid {border} !important;
        background-color: {card_bg} !important;
    }}
    
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    
    .stAlert, .stAlert * {{
        color: inherit !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

inject_custom_css()

# --- Caching ---
@st.cache_resource
def get_ranker():
    return HybridRanker()

@st.cache_data
def get_candidates(file_path):
    return load_candidates(file_path)

# --- PAGES ---

def page_ranker():
    st.markdown("<h1 class='retro-title'>Candidate Ranker Pro</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Enterprise-grade precision matching powered by LightGBM & Deep Semantic Embeddings</div><br>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Theme")
        theme_sel = st.radio("Mode", ["Light", "Dark"], index=0 if st.session_state.theme == 'light' else 1, horizontal=True)
        if theme_sel.lower() != st.session_state.theme:
            st.session_state.theme = theme_sel.lower()
            # Clear particles div to allow recreation with new colors
            js_clear = """<script>
            const p = parent.document.getElementById('particles-js');
            if (p) p.remove();
            </script>"""
            st.components.v1.html(js_clear, height=0, width=0)
            st.rerun()
            
        st.markdown("---")
        st.markdown("### Configuration")
        top_n = st.slider("Top N candidates to display", min_value=5, max_value=50, value=15)
        
        has_api_key = bool(os.getenv('GROQ_API_KEY'))
        use_llm = st.checkbox("Enable LLM Reasoning", value=has_api_key, disabled=not has_api_key)
        
        st.markdown("---")
        st.markdown("### Data Source")
        dataset_choice = st.radio(
            "Select Candidates Dataset:",
            ("Sample Data (50 rows)", "Full Data (100k rows)"),
            index=0 if 'sample' in st.session_state.dataset_path else 1
        )
        if dataset_choice == "Sample Data (50 rows)":
            st.session_state.dataset_path = 'data/raw/sample_candidates.json'
        else:
            st.session_state.dataset_path = 'data/raw/candidates.jsonl'

    # JD
    try:
        if st.session_state.jd_text == "Data Scientist with Python and SQL":
            st.session_state.jd_text = load_job_description('data/raw/job_description.docx')
    except:
        pass

    with st.expander("View / Edit Job Description", expanded=False):
        jd_input = st.text_area("Job Description Details", value=st.session_state.jd_text, height=350, label_visibility="collapsed")
        st.session_state.jd_text = jd_input

    st.markdown("<br>", unsafe_allow_html=True)
    col_spacer1, col_btn, col_spacer2 = st.columns([1, 1.5, 1])
    with col_btn:
        run_btn = st.button("Analyze & Rank Candidates", use_container_width=True)

    results_placeholder = st.empty()

    if run_btn:
        with st.spinner("Analyzing profiles and computing scores..."):
            start_time = time.time()
            candidates = get_candidates(st.session_state.dataset_path)
            st.session_state.raw_candidates = candidates
            ranker = get_ranker()
            
            df = ranker.rank(candidates, jd_input, top_n=top_n)
            
            if use_llm:
                with st.spinner("Generating reasoning templates..."):
                    df = ranker.generate_reasoning(df)
            else:
                df['reasoning'] = ""
                
            st.session_state.ranked_df = df
            elapsed = time.time() - start_time
            
            with results_placeholder.container():
                st.success(f'Successfully processed {len(candidates):,} candidates in {elapsed:.2f} seconds.')
                
    # Render results
    if st.session_state.ranked_df is not None:
        df = st.session_state.ranked_df
        st.markdown("### Ranking Overview")
        m1, m2, m3 = st.columns(3)
        m1.metric("Highest Match Score", f"{df['final_score'].max():.4f}")
        m2.metric("Avg Top-N Score", f"{df['final_score'].mean():.4f}")
        m3.metric("Total Profiles Analyzed", f"{len(st.session_state.raw_candidates) if st.session_state.raw_candidates else len(df):,}")
        
        st.markdown("<br>### Top Recommendations", unsafe_allow_html=True)
        
        # Add CSV export for the hackathon submission
        from src.output_writer import write_submission
        import os
        
        team_id = st.text_input("Enter your Team ID for submission:", value="team_default")
        safe_team_id = "".join([c for c in team_id if c.isalnum() or c in ['_', '-']])
        if not safe_team_id:
            safe_team_id = "team_default"
            
        csv_filename = f"{safe_team_id}.csv"
        
        write_submission(df, f'outputs/{csv_filename}')
        if os.path.exists(f'outputs/{csv_filename}'):
            with open(f'outputs/{csv_filename}', 'rb') as f:
                st.download_button(
                    label="📥 Download Hackathon Submission CSV",
                    data=f,
                    file_name=csv_filename,
                    mime="text/csv",
                    use_container_width=True
                )
        
        # DYNAMIC ACCORDION LIST
        import html as html_lib
        html = "<div style='margin-bottom: 2rem;'>"
        for idx, row in df.iterrows():
            safe_title = html_lib.escape(str(row['current_title']))
            
            html += f"""
            <details class='candidate-row'>
                <summary>
                    <div class='summary-rank'>#{row['rank']}</div>
                    <div class='summary-name'>{safe_title} <span style='font-size:0.85rem; opacity:0.6; font-weight:400;'>(ID: {row['candidate_id']})</span></div>
                    <div class='summary-score'>Match: {row['final_score']:.3f}</div>
                </summary>
                <div class='details-content'>
                    <div class='score-grid'>
                        <div class='score-box'><span class='score-label'>Skills</span><span class='score-value'>{row['skill_match_score']:.2f}</span></div>
                        <div class='score-box'><span class='score-label'>Exp</span><span class='score-value'>{row['experience_score']:.2f}</span></div>
                        <div class='score-box'><span class='score-label'>Edu</span><span class='score-value'>{row['education_score']:.2f}</span></div>
                        <div class='score-box'><span class='score-label'>Trajectory</span><span class='score-value'>{row['trajectory_score']:.2f}</span></div>
                        <div class='score-box'><span class='score-label'>Platform</span><span class='score-value'>{row['platform_signal_score']:.2f}</span></div>
                    </div>
            """
            if row.get('reasoning'):
                safe_reasoning = html_lib.escape(str(row['reasoning']))
                html += f"""
                    <div class='card-reasoning' style='margin-top: 15px;'>
                        <span style='color: #e56b40; font-weight: bold;'>AI Reasoning:</span> {safe_reasoning}
                    </div>
                """
            html += """
                </div>
            </details>
            """
        html += "</div>"
        
        # Streamlit's markdown parser breaks if there are blank lines in raw HTML blocks.
        # Removing all newlines guarantees the entire accordion is parsed as one valid HTML node.
        clean_html = html.replace('\n', '')
        st.markdown(clean_html, unsafe_allow_html=True)
            
        csv_path = 'outputs/ranked_streamlit.csv'
        write_submission(df, csv_path)
        if os.path.exists(csv_path):
            with open(csv_path, 'rb') as f:
                st.download_button("Download Results CSV", data=f.read(), file_name='ranked_candidates.csv', mime='text/csv')

def page_fairness():
    st.markdown("<h1 class='retro-title'>Fairness & Bias Audit</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Automated structural demographic analysis on ranking outcomes.</div><br>", unsafe_allow_html=True)
    
    if st.session_state.ranked_df is None or st.session_state.raw_candidates is None:
        st.warning("Please run the Ranker Pipeline first.")
        return
        
    df = st.session_state.ranked_df
    raw_candidates = st.session_state.raw_candidates
    
    st.markdown("### Demographic Exposure Report")
    
    try:
        from src.fairness_audit import compute_exposure
        with st.spinner("Computing exposure metrics..."):
            summary_stats = compute_exposure(df, raw_candidates, top_n=min(50, len(df)))
            
        # summary_stats format is: {'male': {'pool_count': ..., 'top_n_count': ..., 'pct_of_top_n': ..., 'avg_exposure': ...}, ...}
        
        metrics_list = []
        # Calculate full pool size from summary logic
        total_pool = sum([data['pool_count'] for gen, data in summary_stats.items()])
        
        for gender, data in summary_stats.items():
            pool_pct = (data['pool_count'] / total_pool * 100) if total_pool > 0 else 0
            metrics_list.append({
                'Demographic': gender.capitalize(),
                'Top N Exposure (%)': data['pct_of_top_n'],
                'Baseline (%)': pool_pct
            })
            
        metric_df = pd.DataFrame(metrics_list)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top N Exposure (%)")
            st.bar_chart(metric_df.set_index('Demographic')['Top N Exposure (%)'], color="#e56b40")
        with c2:
            st.markdown("#### Baseline Distribution (%)")
            st.bar_chart(metric_df.set_index('Demographic')['Baseline (%)'], color="#6c757d")
            
    except Exception as e:
        st.error(f"Failed to run fairness audit: {e}")

def page_analytics():
    st.markdown("<h1 class='retro-title'>Model Analytics</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Deep dive into LightGBM feature importance and system ablation.</div><br>", unsafe_allow_html=True)
    
    st.markdown("### LightGBM Feature Importance")
    try:
        model = load_model('data/processed/ltr_model.pkl')
        importances = model.feature_importance(importance_type='gain')
        imp_df = pd.DataFrame({'Feature': FEATURE_NAMES, 'Gain Importance': importances})
        imp_df = imp_df.sort_values(by='Gain Importance', ascending=False)
        st.bar_chart(imp_df.set_index('Feature')['Gain Importance'], color="#e56b40")
    except Exception as e:
        st.error(f"Could not load LTR model for feature importance: {e}")
        
    st.markdown("---")
    st.markdown("### Pipeline Ablation Impact")
    ablation_data = pd.DataFrame({
        'Component': ['Baseline (BGE-Small)', 'Fine-tuned Semantic Embeddings', 'Full LightGBM LTR'],
        'NDCG@10': [0.7816, 0.9233, 0.8689]
    }).set_index('Component')
    st.bar_chart(ablation_data, color="#802f2d")

def page_deep_dive():
    st.markdown("<h1 class='retro-title'>Candidate Deep Dive</h1>", unsafe_allow_html=True)
    st.markdown("<div class='retro-subtitle'>Detailed structural breakdown of individual candidate scores.</div><br>", unsafe_allow_html=True)
    
    if st.session_state.ranked_df is None:
        st.warning("Please run the Ranker Pipeline first.")
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
        
        score_df = pd.DataFrame({
            'Category': ['Skills', 'Experience', 'Education', 'Trajectory', 'Platform', 'Semantic (Context)'],
            'Score': [row['skill_match_score'], row['experience_score'], row['education_score'], row['trajectory_score'], row['platform_signal_score'], row.get('semantic_score', 0)]
        })
        st.bar_chart(score_df.set_index('Category')['Score'], color="#e56b40")
        
        st.markdown("#### Profile Summary")
        summary_text = row.get('profile_summary', 'N/A')
        bg_col = "rgba(0,0,0,0.02)" if st.session_state.theme == 'light' else "rgba(255,255,255,0.02)"
        bord_col = "rgba(0,0,0,0.06)" if st.session_state.theme == 'light' else "rgba(255,255,255,0.06)"
        st.markdown(f"<div style='font-family: \"DM Sans\", sans-serif; font-size: 1.05rem; line-height: 1.7; background: {bg_col}; padding: 1.5rem; border-radius: 12px; border: 1px solid {bord_col};'>{summary_text}</div>", unsafe_allow_html=True)

# --- ROUTER ---
pages = {
    "Ranker Pipeline": page_ranker,
    "Candidate Deep Dive": page_deep_dive,
    "Fairness Audit": page_fairness,
    "Model Analytics": page_analytics
}

with st.sidebar:
    st.markdown("### Navigation")
    selection = st.radio("Go to:", list(pages.keys()), label_visibility="collapsed")

pages[selection]()
