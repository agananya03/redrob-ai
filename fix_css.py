import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

new_css_func = '''def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.cdnfonts.com/css/chillax');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background-color: #ffffff;
        color: #1a1a1a;
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Typography */
    .retro-title {
        font-family: 'Chillax', system-ui, sans-serif !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        color: #1a1a1a !important;
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    
    .retro-subtitle {
        font-family: 'DM Sans', sans-serif !important;
        color: #e56b40 !important;
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        margin-top: 0px !important;
    }
    
    /* Subheaders */
    h1, h2, h3, .st-emotion-cache-1629p8f h2 {
        font-family: 'Chillax', system-ui, sans-serif !important;
        color: #1a1a1a !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f8f4ed !important;
        border-right: 1px solid #e5e7eb;
    }
    [data-testid="stSidebar"] * {
        font-family: 'DM Sans', sans-serif !important;
        color: #1a1a1a !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #1a1a1a !important;
        background-color: #ffffff;
        border-radius: 1.125rem;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 1px 2px rgb(0 0 0 / 0.03);
    }
    div[data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
    }
    div[data-testid="stExpander"] > div:nth-child(2) {
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-top: none;
        border-radius: 0 0 1.125rem 1.125rem;
        background: #ffffff;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgb(0 0 0 / 0.04), 0 4px 14px rgb(0 0 0 / 0.05);
    }
    
    /* Text Inputs */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #1a1a1a !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.625rem !important;
        padding: 1rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1rem !important;
        transition: all 0.2s ease;
    }
    .stTextArea textarea:focus {
        border-color: #e56b40 !important;
        box-shadow: 0 0 0 3px rgba(229, 107, 64, 0.2) !important;
    }
    
    /* Primary Buttons */
    div.stButton > button {
        background: #e56b40 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.625rem !important;
        padding: 0.75rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 1px 3px rgb(0 0 0 / 0.04), 0 4px 14px rgb(0 0 0 / 0.05) !important;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgb(0 0 0 / 0.07), 0 12px 28px rgb(0 0 0 / 0.08) !important;
        background: #f95955 !important;
    }
    div.stButton > button:active {
        transform: translateY(0);
    }
    
    /* Metrics Container */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        padding: 1.25rem;
        border-radius: 1.125rem;
        box-shadow: 0 1px 3px rgb(0 0 0 / 0.04), 0 4px 14px rgb(0 0 0 / 0.05);
        transition: transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgb(0 0 0 / 0.07), 0 12px 28px rgb(0 0 0 / 0.08);
    }
    [data-testid="stMetricValue"] {
        font-family: 'Chillax', system-ui, sans-serif !important;
        color: #e56b40 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #6c757d !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.85rem !important;
    }
    
    /* DataFrame */
    [data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 1.125rem;
        border: 1px solid rgba(0, 0, 0, 0.06);
        padding: 10px;
        box-shadow: 0 1px 3px rgb(0 0 0 / 0.04), 0 4px 14px rgb(0 0 0 / 0.05);
        font-family: 'DM Sans', sans-serif !important;
    }
    
    /* Hide top header line */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Streamlit Alert/Success boxes */
    .stAlert {
        border-radius: 0.625rem !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)'''

new_content = re.sub(r'def inject_custom_css\(\):.*?(?=inject_custom_css\(\)\n)', new_css_func + '\n\n', content, flags=re.DOTALL)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("CSS successfully replaced with true iitm-5 light mode theme!")
