import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Noida Energy Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- THEME DETECTION ----------------
theme = st.get_option("theme.base") or "light"
is_dark = theme == "dark"

COLORS_THEME = {
    "bg": "#0E1117" if is_dark else "#FFFFFF",
    "secondary_bg": "#161B22" if is_dark else "#F8FFFE",
    "text": "#E6EDF3" if is_dark else "#1A1A1A",
    "subtext": "#9DA7B3" if is_dark else "#5F5E5A",
    "border": "#30363D" if is_dark else "#E0E0E0",
    "grid": "#2A2F36" if is_dark else "#F0F0F0",
}

# Plotly auto theme
pio.templates.default = "plotly_dark" if is_dark else "plotly_white"

# ---------------- FULL CSS FIX ----------------
st.markdown(f"""
<style>
/* Modern CSS Custom Properties */
:root {{
    --primary: #1D9E75;
    --primary-light: #2bc194;
    --primary-dark: #0F6E56;
    --bg-color: {COLORS_THEME["bg"]};
    --card-bg: {COLORS_THEME["secondary_bg"]};
    --text-main: {COLORS_THEME["text"]};
    --text-muted: {COLORS_THEME["subtext"]};
    --border-color: {COLORS_THEME["border"]};
    --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-md: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-lg: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    --radius-md: 16px;
    --radius-lg: 24px;
}}

html, body, .stApp {{
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}}

/* Sidebar Styling */
section[data-testid="stSidebar"] {{
    background-color: var(--card-bg) !important;
    border-right: 1px solid var(--border-color);
}}

section[data-testid="stSidebar"] * {{
    color: var(--text-main) !important;
}}

/* Inputs & Form Elements */
input, textarea, div[data-baseweb="select"] > div {{
    background-color: var(--bg-color) !important;
    color: var(--text-main) !important;
    border-radius: 8px !important;
    border: 1px solid var(--border-color) !important;
    transition: all 0.3s ease;
}}

input:focus, div[data-baseweb="select"] > div:focus-within {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(29, 158, 117, 0.2) !important;
}}

button[data-baseweb="tab"] {{
    color: var(--text-muted) !important;
    font-weight: 600;
    transition: color 0.3s ease;
}}

button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--primary) !important;
}}

/* Premium Metric Cards */
.metric-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}}

.metric-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--primary), var(--primary-light));
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.metric-card:hover {{
    transform: translateY(-5px);
    box-shadow: var(--shadow-md);
    border-color: var(--primary-light);
}}

.metric-card:hover::before {{
    opacity: 1;
}}

.metric-card .val {{
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}}

.metric-card .lbl {{
    color: var(--text-muted);
    font-size: 0.95rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.metric-card .sub {{
    font-size: 0.8rem;
    color: var(--text-muted);
    opacity: 0.8;
    margin-top: 0.25rem;
}}

/* Main Header Styling */
.main-header {{
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: var(--radius-lg);
    margin-bottom: 2.5rem;
    box-shadow: var(--shadow-md);
    text-align: center;
    position: relative;
    overflow: hidden;
}}

.main-header::after {{
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    pointer-events: none;
}}

.main-header h1 {{
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
    color: white !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}

.main-header p {{
    font-size: 1.1rem;
    opacity: 0.9;
    max-width: 800px;
    margin: 0 auto;
}}

/* Section Titles */
.section-title {{
    color: var(--text-main);
    font-size: 1.4rem;
    font-weight: 700;
    margin: 2rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 10px;
}}

.section-title::before {{
    content: '';
    display: block;
    width: 6px;
    height: 24px;
    background: var(--primary);
    border-radius: 4px;
}}

/* Insight Boxes */
.insight-box {{
    background: {"#0F2E2A" if is_dark else "linear-gradient(to right, #E1F5EE, #F8FFFE)"};
    color: var(--text-main);
    border-left: 5px solid var(--primary);
    padding: 1.5rem;
    border-radius: 0 var(--radius-md) var(--radius-md) 0;
    margin: 1.5rem 0;
    box-shadow: var(--shadow-sm);
    font-size: 1.05rem;
    line-height: 1.6;
}}

/* DataFrames / Tables */
[data-testid="stDataFrame"] {{
    background-color: var(--card-bg);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border-color);
}}

/* Footer */
.footer {{
    color: var(--text-muted);
    border-top: 1px solid var(--border-color);
    padding-top: 2rem;
    margin-top: 4rem;
    text-align: center;
    font-size: 0.9rem;
    font-weight: 500;
}}
</style>
""", unsafe_allow_html=True)


# ---------------- FIXED PLOTLY ----------------
PLOTLY_LAYOUT = dict(
    plot_bgcolor=COLORS_THEME["bg"],
    paper_bgcolor=COLORS_THEME["bg"],
    font=dict(color=COLORS_THEME["text"]),
    margin=dict(t=30, b=40, l=50, r=20),
    xaxis=dict(gridcolor=COLORS_THEME["grid"], linecolor=COLORS_THEME["border"]),
    yaxis=dict(gridcolor=COLORS_THEME["grid"], linecolor=COLORS_THEME["border"]),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig
    
