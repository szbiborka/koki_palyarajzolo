# =============================================================================
# UI ASSETS - CSS formázások, SVG ikonok és vizuális segédfüggvények
# =============================================================================
import streamlit as st

def setup_css():
    """Befecskendezi a globális CSS stílusokat a Streamlit felületbe."""
    st.markdown("""
    <style>
        :root {
            --cream: #FAF9F6;
            --sage-tint: #EBEFDF;       
            --border-sage: #C7D1B5;     
            --sage-light: #A4B595;
            --sage: #6B8059;
            --sage-deep: #3A4A28;
            --rosewood-tint: #F7EAE8;
            --rosewood-light: #D4A39F;
            --rosewood: #8A4F4F;
            --rosewood-deep: #542D2D;
            --taupe: #948F7F;
            --taupe-tint: #F2F0EB;
            --taupe-deep: #5E5A4A;
            --neural-highlight: #D96C75;
            --border-radius-soft: 16px;
            --border-radius-pill: 24px;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        }

        .stApp { 
            background-color: var(--cream); 
            background-image: radial-gradient(var(--taupe-tint) 1px, transparent 1px);
            background-size: 20px 20px;
        }

        [data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
            background-color: var(--sage-tint);
            border-right: 1px dashed var(--border-sage);
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }

        .sidebar-title {
            font-size: 1.35rem; font-weight: 800; letter-spacing: 0.06em; text-transform: uppercase;
            color: var(--sage-deep); margin-bottom: 0.2rem; display: flex; align-items: center;
            font-family: 'Georgia', serif; 
        }
        .sidebar-subtitle {
            font-size: 0.75rem; color: var(--rosewood); letter-spacing: 0.1em;
            text-transform: uppercase; margin-bottom: 1.5rem; font-weight: 600;
        }

        .result-card {
            background: #FFFFFF; border: 1px solid rgba(199, 209, 181, 0.4);
            border-left: 5px solid var(--sage); border-radius: var(--border-radius-soft);
            padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
            box-shadow: 0 4px 12px rgba(95, 115, 80, 0.04), 0 1px 3px rgba(0, 0, 0, 0.02);
            transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
            position: relative; overflow: hidden;
        }
        .result-card::before {
            content: ""; position: absolute; top: -15px; right: -15px; width: 60px; height: 60px;
            background: radial-gradient(circle, var(--sage-tint) 10%, transparent 10%),
                        radial-gradient(circle, var(--sage-tint) 10%, transparent 10%);
            background-size: 10px 10px; opacity: 0.5; border-radius: 50%;
        }
        .result-card:hover {
            transform: translateY(-3px); box-shadow: 0 10px 20px rgba(95, 115, 80, 0.08); border-color: var(--sage-light);
        }
        .result-card.positive { border-left-color: var(--sage); }
        .result-card.negative { border-left-color: var(--taupe); opacity: 0.9; }
        .result-card.filtered-out {
            border-left-color: var(--rosewood-light);
            background: linear-gradient(to right, var(--rosewood-tint), #FFFFFF);
        }
        .result-card h4 {
            margin: 0 0 0.6rem 0; font-size: 1.1rem; font-weight: 700; color: var(--sage-deep); font-family: 'Georgia', serif;
        }
        .result-card .meta { font-size: 0.85rem; color: var(--taupe-deep); font-weight: 500; }

        .tag-yes, .tag-no, .tag-filtered {
            display: inline-block; font-size: 0.65rem; font-weight: 800; letter-spacing: 0.08em;
            padding: 0.3rem 0.8rem; border-radius: var(--border-radius-pill);
            text-transform: uppercase; margin-right: 0.6rem; margin-bottom: 0.5rem;
            box-shadow: inset 0 0 0 1px rgba(0,0,0,0.05);
        }
        .tag-yes { background: var(--sage-tint); color: var(--sage-deep); }
        .tag-no { background: var(--taupe-tint); color: var(--taupe-deep); }
        .tag-filtered { background: rgba(212, 163, 159, 0.25); color: var(--rosewood-deep); }

        .page-header { margin-bottom: 2.5rem; }
        .page-header h1 { font-size: 2.2rem; font-weight: 800; letter-spacing: 0.02em; color: var(--sage-deep); margin: 0; font-family: 'Georgia', serif; }
        .page-header p { font-size: 0.95rem; color: var(--rosewood); font-weight: 600; margin: 0.4rem 0 0 0; letter-spacing: 0.05em; text-transform: uppercase; }

        hr {
            border: none; height: 2px; background: linear-gradient(to right, transparent, var(--border-sage), transparent);
            position: relative; margin: 3rem 0; overflow: visible;
        }
        hr::after {
            content: "●"; color: var(--neural-highlight); font-size: 18px; position: absolute;
            left: 50%; top: -12px; transform: translateX(-50%); text-shadow: 0 0 8px rgba(217, 108, 117, 0.4);
        }

        .stButton > button { 
            border-radius: var(--border-radius-pill) !important; font-weight: 700 !important; 
            letter-spacing: 0.05em !important; text-transform: uppercase; font-size: 0.85rem !important;
            transition: all 0.25s ease !important; padding: 0.5rem 1.5rem !important;
        }
        [data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, var(--rosewood), var(--rosewood-light)) !important;
            border: none !important; color: #FFFFFF !important; box-shadow: 0 4px 10px rgba(138, 79, 79, 0.3) !important;
        }
        [data-testid="baseButton-primary"]:hover {
            transform: translateY(-2px); box-shadow: 0 6px 15px rgba(138, 79, 79, 0.4) !important;
            background: linear-gradient(135deg, var(--rosewood-deep), var(--rosewood)) !important;
        }
        [data-testid="baseButton-secondary"] { border: 1px solid var(--border-sage) !important; color: var(--sage-deep) !important; background-color: transparent !important; }
        [data-testid="baseButton-secondary"]:hover { border-color: var(--sage) !important; background-color: var(--sage-tint) !important; }

        [data-testid="stTabs"] button { font-weight: 700; color: var(--taupe); letter-spacing: 0.02em; padding-bottom: 0.8rem; }
        [data-testid="stTabs"] button[aria-selected="true"] { color: var(--sage-deep); border-bottom-color: var(--neural-highlight); border-bottom-width: 3px; }

        [data-testid="stMetricValue"] { color: var(--sage-deep); font-weight: 800; font-family: 'Georgia', serif; }
        [data-testid="stMetricLabel"] { color: var(--rosewood); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }

        footer, #MainMenu { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

NEURON_MARK = """
<svg width="24" height="24" viewBox="0 0 44 44" xmlns="http://www.w3.org/2000/svg" style="vertical-align:-6px;margin-right:8px;">
    <g stroke="#5F7350" stroke-width="2" fill="none" stroke-linecap="round">
        <path d="M22 17 Q18 8 12 6"/><path d="M22 17 Q26 7 33 9"/><path d="M27 22 Q36 20 40 14"/>
        <path d="M27 25 Q37 28 41 35"/><path d="M17 27 Q12 35 6 38"/><path d="M17 22 Q7 21 3 16"/>
    </g>
    <g fill="#5F7350">
        <circle cx="12" cy="6" r="2"/><circle cx="33" cy="9" r="2"/><circle cx="40" cy="14" r="2"/>
        <circle cx="41" cy="35" r="2"/><circle cx="6" cy="38" r="2"/><circle cx="3" cy="16" r="2"/>
    </g>
    <circle cx="22" cy="22" r="5" fill="#7A3B3B"/>
</svg>
"""

SYNAPSE_MARK = """
<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align:-1px;margin-right:6px;">
    <circle cx="12" cy="12" r="7" stroke="#7A3B3B" stroke-width="3"/>
    <circle cx="12" cy="12" r="3" fill="#5F7350"/>
</svg>
"""

def section_header(title: str):
    """Generates a custom formatted subheader with the SYNAPSE_MARK icon."""
    st.markdown(f"<h4>{SYNAPSE_MARK}{title}</h4>", unsafe_allow_html=True)