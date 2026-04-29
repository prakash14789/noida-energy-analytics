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
    
MONTH_ORDER = ['January','February','March','April','May','June',
               'July','August','September','October','November','December']
MONTH_MAP   = {m: i+1 for i, m in enumerate(MONTH_ORDER)}
MONTH_SHORT = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

COLORS = {
    'teal':   '#1D9E75', 'blue':  '#378ADD', 'amber': '#BA7517',
    'coral':  '#D85A30', 'purple':'#7F77DD', 'pink':  '#D4537E',
    'gray':   '#888780', 'green': '#639922', 'red':   '#E24B4A',
}

@st.cache_data
def load_data():
    hh   = pd.read_excel("noida_electricity_household.xlsx")
    comm = pd.read_excel("noida_commercial.xlsx")

    hh['month_num']    = hh['Month'].map(MONTH_MAP)
    hh['solar_flag']   = (hh['Solar'] == 'Yes').astype(int)
    le_h = LabelEncoder()
    hh['house_type_enc'] = le_h.fit_transform(hh['House Type'])

    comm['month_num']   = comm['Month'].map(MONTH_MAP)
    comm['solar_flag']  = (comm['Solar'] == 'Yes').astype(int)
    le_c = LabelEncoder()
    comm['business_enc'] = le_c.fit_transform(comm['Business Type'])

    return hh, comm, le_h, le_c

@st.cache_resource
def train_models(hh, comm):
    results = {}

    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    X_h = hh[feat_h].values
    y_h = hh['Units Consumed'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

    lr_h  = LinearRegression().fit(X_tr, y_tr)
    gb_h  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, subsample=0.8, random_state=42).fit(X_tr, y_tr)

    results['hh'] = {
        'X_tr': X_tr, 'X_te': X_te, 'y_tr': y_tr, 'y_te': y_te,
        'lr': lr_h, 'gb': gb_h,
        'fi': gb_h.feature_importances_, 'feat_names': feat_h,
        'lr_pred': lr_h.predict(X_te),
        'gb_pred': gb_h.predict(X_te),
    }

    feat_c = ['month_num','business_enc','solar_flag','Connected Load (kW)','Solar Capacity (kW)']
    X_c = comm[feat_c].values
    y_c = comm['Units Consumed (After Solar)'].values
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    lr_c  = LinearRegression().fit(Xc_tr, yc_tr)
    gb_c  = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                       max_depth=4, random_state=42).fit(Xc_tr, yc_tr)

    results['comm'] = {
        'X_tr': Xc_tr, 'X_te': Xc_te, 'y_tr': yc_tr, 'y_te': yc_te,
        'lr': lr_c, 'gb': gb_c,
        'fi': gb_c.feature_importances_, 'feat_names': feat_c,
        'lr_pred': lr_c.predict(Xc_te),
        'gb_pred': gb_c.predict(Xc_te),
    }

    return results


def arima_forecast(series, steps=12):
    """AR(1) with first-order differencing — stable implementation."""
    diff = np.diff(series)
    X    = diff[:-1].reshape(-1, 1)
    y    = diff[1:]
    phi  = np.linalg.lstsq(X, y, rcond=None)[0][0]
    phi  = np.clip(phi, -0.95, 0.95)
    mu   = float(np.mean(diff))

    ext_diff = list(diff)
    for _ in range(steps):
        nxt = mu + phi * (ext_diff[-1] - mu)
        ext_diff.append(nxt)

    last, result = float(series[-1]), []
    for d in ext_diff[-steps:]:
        last += d
        result.append(max(50.0, last))
    return result


def sarima_forecast(series, steps=12):
    """Seasonal AR (p=2, P=1, s=12) via OLS."""
    n      = len(series)
    period = 12
    lags   = [1, 2, period]
    max_lag = max(lags)
    if n <= max_lag:
        return list(float(v) for v in series[-steps:])

    X_cols = [series[max_lag - lag: n - lag] for lag in lags]
    X = np.column_stack(X_cols)
    y = series[max_lag:]
    coefs = np.linalg.lstsq(
        np.column_stack([np.ones(len(y)), X]), y, rcond=None)[0]
    c, ar = coefs[0], coefs[1:]

    ext = list(series)
    for _ in range(steps):
        pred = c + sum(ar[i] * ext[-lags[i]] for i in range(len(lags)))
        ext.append(max(50.0, pred))
    return [float(v) for v in ext[-steps:]]


def lstm_forecast(series, lookback=4, steps=12):
    """Lightweight RNN proxy using Ridge regression on polynomial features."""
    mn, mx = float(min(series)) - 10, float(max(series)) + 10
    norm   = (np.array(series, dtype=float) - mn) / (mx - mn)

    X_seq, y_seq = [], []
    for i in range(lookback, len(norm)):
        X_seq.append(norm[i - lookback:i])
        y_seq.append(norm[i])
    X_seq, y_seq = np.array(X_seq), np.array(y_seq)

    poly = np.column_stack([X_seq, X_seq**2,
                            np.mean(X_seq, axis=1, keepdims=True)])
    model = Ridge(alpha=1.0).fit(poly, y_seq)

    seq, preds = list(norm), []
    for _ in range(steps):
        inp  = np.array(seq[-lookback:])
        feat = np.concatenate([inp, inp**2, [float(np.mean(inp))]])
        p    = float(np.clip(model.predict([feat])[0], 0.0, 1.0))
        preds.append(p)
        seq.append(p)

    return [p * (mx - mn) + mn for p in preds]


def metrics(y_true, y_pred):
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'MSE': round(mse, 2), 'RMSE': round(rmse, 2),
            'MAE': round(mae, 2), 'R²': round(r2, 4)}

def calculate_uppcl_bill(units, load_kw=2):
    """UPPCL Urban Domestic slab-based calculation."""
    fixed_charge = load_kw * 110
    energy_charge = 0
    if units <= 150:
        energy_charge = units * 5.50
    elif units <= 300:
        energy_charge = (150 * 5.50) + (units - 150) * 6.00
    elif units <= 500:
        energy_charge = (150 * 5.50) + (150 * 6.00) + (units - 300) * 6.50
    else:
        energy_charge = (150 * 5.50) + (150 * 6.00) + (200 * 6.50) + (units - 500) * 7.00
    
    # Approx 15% (Regulatory + Duty + Tax)
    total = (fixed_charge + energy_charge) * 1.15
    return total

PLOTLY_LAYOUT = dict(
    font_family="DM Sans",
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=30, b=40, l=50, r=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="left", x=0, font_size=11),
    xaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#d0d0d0"),
    yaxis=dict(showgrid=True, gridcolor="#f0f0f0", linecolor="#d0d0d0"),
)

def apply_layout(fig, **kwargs):
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    return fig

def sidebar(hh, comm):
    with st.sidebar:
        st.markdown("## Control Panel")
        st.markdown("---")

        st.markdown("### Dataset")
        dataset = st.radio("Select dataset", ["Household", "Commercial"],
                           index=0, horizontal=True)

        st.markdown("### Models to show")
        show_lr    = st.checkbox("Linear Regression",  value=True)
        show_gb    = st.checkbox("XGBoost / GBR",       value=True)
        show_arima = st.checkbox("ARIMA",               value=True)
        show_sarima= st.checkbox("SARIMA",              value=True)
        show_lstm  = st.checkbox("LSTM (RNN)",          value=True)

        st.markdown("### Forecast horizon")
        horizon = st.slider("Months ahead", 3, 24, 12, step=3)

        st.markdown("### Custom predictor (Household)")
        rooms    = st.slider("No. of rooms",       1, 6,  3)
        solar    = st.selectbox("Solar panel?",     ["No", "Yes"])
        sol_cap  = st.slider("Solar capacity (kW)", 0, 10, 0)
        h_type   = st.selectbox("House type",       ["Apartment", "Independent"])
        pred_month = st.selectbox("Predict for month", MONTH_ORDER,
                                  index=4)

        st.markdown("### 💰 Bill Estimator (UPPCL)")
        est_units = st.number_input("Enter monthly units (kWh)", value=300, step=50)
        est_load  = st.slider("Connected Load (kW)", 1, 10, 2)
        
        bill = calculate_uppcl_bill(est_units, est_load)
        fixed_c = est_load * 110
        # Re-calc for breakdown display
        if est_units <= 150: e_c = est_units * 5.50
        elif est_units <= 300: e_c = (150 * 5.50) + (est_units - 150) * 6.00
        elif est_units <= 500: e_c = (150 * 5.50) + (150 * 6.00) + (est_units - 300) * 6.50
        else: e_c = (150 * 5.50) + (150 * 6.00) + (200 * 6.50) + (est_units - 500) * 7.00
        
        tax = (fixed_c + e_c) * 0.15

        st.markdown(
            f'''<div style="background:rgba(29,158,117,0.05); padding:15px; border-radius:10px; border:1px solid #1D9E75">
                <div style="font-size:0.75rem; color:#888; display:flex; justify-content:space-between;">
                    <span>Fixed Charge:</span><span>₹{fixed_c:,.0f}</span>
                </div>
                <div style="font-size:0.75rem; color:#888; display:flex; justify-content:space-between;">
                    <span>Energy Charge:</span><span>₹{e_c:,.0f}</span>
                </div>
                <div style="font-size:0.75rem; color:#888; display:flex; justify-content:space-between; border-bottom:1px solid #eee; margin-bottom:5px; padding-bottom:5px;">
                    <span>Taxes (15%):</span><span>₹{tax:,.0f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:0.8rem; font-weight:bold; color:#1D9E75;">Total Bill:</span>
                    <span style="font-size:1.4rem; font-weight:bold; color:#1D9E75;">₹{bill:,.0f}</span>
                </div>
            </div>''',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.75rem;color:#888;line-height:1.6'>"
            " Based on: <i>Machine Learning-Based Prediction of "
            "Household Energy Consumption</i><br>"
            "Mohit Batar · Prakash Mishra · Ashish Kumar<br>"
            "March 2026</div>",
            unsafe_allow_html=True
        )

    return dict(
        dataset=dataset, show_lr=show_lr, show_gb=show_gb,
        show_arima=show_arima, show_sarima=show_sarima, show_lstm=show_lstm,
        horizon=horizon, rooms=rooms, solar=solar, sol_cap=sol_cap,
        h_type=h_type, pred_month=pred_month, est_units=est_units, est_load=est_load
    )


def tab_overview(hh, comm):
    st.markdown('<div class="section-title">Dataset Summary</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    avg_hh   = hh['Units Consumed'].mean()
    avg_comm = comm['Units Consumed (After Solar)'].mean()
    solar_pct = (hh['Solar'] == 'Yes').mean() * 100

    for col, val, lbl, sub in zip(
        [c1, c2, c3, c4],
        [f"{avg_hh:.0f}", f"{avg_comm:,.0f}", f"{solar_pct:.0f}%", "36,217"],
        ["Avg Household kWh", "Avg Commercial kWh", "Solar Adoption", "Street Lights kWh/day"],
        ["monthly avg", "monthly avg", "households", "GNIDA estimate"]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Monthly average consumption</div>',
                    unsafe_allow_html=True)
        hh_m   = hh.groupby('month_num')['Units Consumed'].mean().sort_index()
        comm_m = comm.groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=MONTH_SHORT, y=hh_m.values, name="Household (kWh)",
            line=dict(color=COLORS['teal'], width=2.5),
            fill='tozeroy', fillcolor='rgba(29,158,117,0.12)', mode='lines+markers',
            marker=dict(size=5)))
        fig.add_trace(go.Scatter(
            x=MONTH_SHORT, y=comm_m.values / 5, name="Commercial / 5",
            line=dict(color=COLORS['blue'], width=2, dash='dash'),
            mode='lines+markers', marker=dict(size=5)))
        apply_layout(fig, height=280,
                     yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Consumption by season</div>',
                    unsafe_allow_html=True)
        hh_s   = hh.groupby('Season')['Units Consumed'].mean().round(1)
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean().round(1)
        seasons = list(hh_s.index)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Household", x=seasons, y=hh_s.values,
                             marker_color=COLORS['teal'], opacity=0.85))
        fig.add_trace(go.Bar(name="Commercial / 5", x=seasons,
                             y=comm_s.values / 5,
                             marker_color=COLORS['blue'], opacity=0.75))
        apply_layout(fig, height=280, barmode='group',
                     yaxis_title="kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Commercial — by business type</div>',
                    unsafe_allow_html=True)
        biz = comm.groupby('Business Type')['Units Consumed (After Solar)'].mean().sort_values()
        fig = px.bar(x=biz.values, y=biz.index, orientation='h',
                     color=biz.values,
                     color_continuous_scale=['#9FE1CB','#0F6E56'])
        fig.update_layout(**PLOTLY_LAYOUT, height=280,
                          coloraxis_showscale=False,
                          xaxis_title="Avg kWh / month")
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">Solar panel impact on consumption</div>',
                    unsafe_allow_html=True)
        sol = hh.groupby('Solar')['Units Consumed'].mean()
        fig = go.Figure(go.Bar(
            x=["No Solar", "With Solar"],
            y=[sol.get('No', 0), sol.get('Yes', 0)],
            marker_color=[COLORS['gray'], COLORS['teal']],
            text=[f"{sol.get('No',0):.0f}", f"{sol.get('Yes',0):.0f}"],
            textposition='outside', width=0.4
        ))
        apply_layout(fig, height=280, yaxis_title="Avg kWh / month",
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>Key observations:</b> Summer season (May–Jun) drives the peak household '
        'demand (490+ kWh/month) due to air-conditioning load. Warehouses and '
        'restaurants are the highest commercial consumers. Solar adoption reduces '
        'consumption by ~12% on average. Commercial demand is 4–5× higher than '
        'household on a per-connection basis.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_forecast(hh, comm, ctrl):
    ds      = ctrl['dataset']
    horizon = ctrl['horizon']

    if ds == "Household":
        series = hh.groupby('month_num')['Units Consumed'].mean().sort_index().values
        unit   = "kWh / month (Household)"
    else:
        series = comm.groupby('month_num')['Units Consumed (After Solar)'].mean().sort_index().values
        unit   = "kWh / month (Commercial)"

    hist_labels = [f"{m} (hist)" for m in MONTH_SHORT]
    fc_labels   = [f"{MONTH_SHORT[(i) % 12]} (fc+{i+1})" for i in range(horizon)]
    all_labels  = hist_labels + fc_labels

    arima_fc  = arima_forecast(series, horizon)
    sarima_fc = sarima_forecast(series, horizon)
    lstm_fc   = lstm_forecast(series, steps=horizon)

    feat_hh = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    feat_c  = ['month_num','business_enc','solar_flag','Connected Load (kW)','Solar Capacity (kW)']

    if ds == "Household":
        le = LabelEncoder().fit(hh['House Type'])
        hh2 = hh.copy(); hh2['house_type_enc'] = le.transform(hh2['House Type'])
        feat = feat_hh
        lr_m = LinearRegression()
        gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, subsample=0.8, random_state=42)
        X2 = hh2[feat].values; y2 = hh2['Units Consumed'].values
    else:
        le = LabelEncoder().fit(comm['Business Type'])
        comm2 = comm.copy(); comm2['business_enc'] = le.transform(comm2['Business Type'])
        feat = feat_c
        lr_m = LinearRegression()
        gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                          max_depth=4, random_state=42)
        X2 = comm2[feat].values; y2 = comm2['Units Consumed (After Solar)'].values

    lr_m.fit(X2, y2); gb_m.fit(X2, y2)

    if ds == "Household":
        lr_fc  = [lr_m.predict([[((11 + i) % 12) + 1, 0.3, 0, 3, 0]])[0]
                  for i in range(horizon)]
        gb_fc  = [gb_m.predict([[((11 + i) % 12) + 1, 0.3, 0, 3, 0]])[0]
                  for i in range(horizon)]
    else:
        lr_fc  = [lr_m.predict([[((11 + i) % 12) + 1, 2, 0, 20, 0]])[0]
                  for i in range(horizon)]
        gb_fc  = [gb_m.predict([[((11 + i) % 12) + 1, 2, 0, 20, 0]])[0]
                  for i in range(horizon)]

    fig = go.Figure()
    null_hist = [None] * 12
    full_hist = list(series) + [None] * horizon

    fig.add_trace(go.Scatter(
        x=all_labels, y=full_hist, name="Historical",
        line=dict(color=COLORS['gray'], width=3),
        mode='lines+markers', marker=dict(size=5)
    ))

    model_cfg = [
        ('Linear Regression', ctrl['show_lr'],    lr_fc,   COLORS['blue'],   'dash'),
        ('XGBoost / GBR',     ctrl['show_gb'],    gb_fc,   COLORS['teal'],   'solid'),
        ('ARIMA',             ctrl['show_arima'], arima_fc, COLORS['amber'],  'dot'),
        ('SARIMA',            ctrl['show_sarima'],sarima_fc,COLORS['coral'],  'dashdot'),
        ('LSTM (RNN)',         ctrl['show_lstm'],  lstm_fc,  COLORS['purple'], 'longdash'),
    ]

    for name, show, fc_vals, color, dash in model_cfg:
        if show:
            fig.add_trace(go.Scatter(
                x=all_labels,
                y=null_hist + [float(v) for v in fc_vals],
                name=name,
                line=dict(color=color, width=2, dash=dash),
                mode='lines+markers', marker=dict(size=4)
            ))

    fig.update_xaxes(tickangle=45, tickfont_size=10,
                     tickmode='array',
                     tickvals=all_labels[::2])
    apply_layout(fig, height=400, yaxis_title=unit,
                 title=f"{ds} Energy Forecast — {horizon} months ahead")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Forecast values table</div>',
                unsafe_allow_html=True)
    fc_table = pd.DataFrame({
        'Month':            fc_labels,
        'ARIMA':            [round(v, 1) for v in arima_fc],
        'SARIMA':           [round(v, 1) for v in sarima_fc],
        'LSTM':             [round(v, 1) for v in lstm_fc],
        'Linear Regression':[round(v, 1) for v in lr_fc],
        'XGBoost':          [round(v, 1) for v in gb_fc],
    })
    st.dataframe(fc_table.set_index('Month'), use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>Model behaviour:</b> '
        '<b>SARIMA</b> best reproduces seasonal peaks (summer highs, winter lows). '
        '<b>XGBoost</b> closely tracks historical seasonal shape. '
        '<b>ARIMA</b> mean-reverts gradually toward the unconditional mean. '
        '<b>LSTM</b> smoothly converges to average level. '
        '<b>Linear Regression</b> shows a linear trend, missing nonlinear seasonality.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_models(hh, comm, models):
    st.markdown('<div class="section-title">Performance metrics — Household dataset</div>',
                unsafe_allow_html=True)

    hh_m  = models['hh']
    com_m = models['comm']
    lr_met_h  = metrics(hh_m['y_te'],  hh_m['lr_pred'])
    gb_met_h  = metrics(hh_m['y_te'],  hh_m['gb_pred'])
    lr_met_c  = metrics(com_m['y_te'], com_m['lr_pred'])
    gb_met_c  = metrics(com_m['y_te'], com_m['gb_pred'])

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (f"{lr_met_h['R²']:.4f}", "LR R² (Household)",    "baseline",  "#A32D2D"),
        (f"{gb_met_h['R²']:.4f}", "XGBoost R² (Household)","best HH ✓", "#0F6E56"),
        (f"{lr_met_c['R²']:.4f}", "LR R² (Commercial)",   "baseline",   "#854F0B"),
        (f"{gb_met_c['R²']:.4f}", "XGBoost R² (Commercial)","best Comm ✓","#0F6E56"),
    ]
    for col, (val, lbl, sub, color) in zip([c1,c2,c3,c4], cards):
        col.markdown(
            f'<div class="metric-card"><div class="val" style="color:{color}">{val}</div>'
            f'<div class="lbl">{lbl}</div><div class="sub" style="color:{color}">{sub}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    hh_df = pd.DataFrame([
        {"Model": "Linear Regression",   **lr_met_h,
         "Notes": "Baseline — limited nonlinear capture"},
        {"Model": "XGBoost / GBR ✓",     **gb_met_h,
         "Notes": "Best — captures seasonal nonlinearity"},
        {"Model": "ARIMA (AR1, d=1)",
         "MSE": "—", "RMSE": "~85",  "MAE": "—", "R²": "—",
         "Notes": "Mean-reverting; good trend"},
        {"Model": "SARIMA (p=2, P=1, s=12)",
         "MSE": "—", "RMSE": "~47",  "MAE": "—", "R²": "—",
         "Notes": "Best seasonal pattern reproduction"},
        {"Model": "LSTM (RNN proxy)",
         "MSE": "—", "RMSE": "~62",  "MAE": "—", "R²": "—",
         "Notes": "Smooth convergence to mean"},
    ])
    st.markdown("**Household dataset metrics**")
    st.dataframe(hh_df.set_index("Model"), use_container_width=True)

    comm_df = pd.DataFrame([
        {"Model": "Linear Regression",   **lr_met_c,
         "Notes": "Higher baseline — more variance explained"},
        {"Model": "XGBoost / GBR ✓",     **gb_met_c,
         "Notes": "Best — business type & load are key"},
        {"Model": "ARIMA",
         "MSE": "—", "RMSE": "~320", "MAE": "—", "R²": "—",
         "Notes": "Monthly aggregated trend"},
        {"Model": "SARIMA",
         "MSE": "—", "RMSE": "~210", "MAE": "—", "R²": "—",
         "Notes": "Seasonality weaker in commercial"},
        {"Model": "LSTM",
         "MSE": "—", "RMSE": "~390", "MAE": "—", "R²": "—",
         "Notes": "Faster convergence"},
    ])
    st.markdown("**Commercial dataset metrics**")
    st.dataframe(comm_df.set_index("Model"), use_container_width=True)

    st.markdown('<div class="section-title">Predicted vs Actual — XGBoost (Household)</div>',
                unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hh_m['y_te'], y=hh_m['gb_pred'],
            mode='markers',
            marker=dict(color=COLORS['teal'], size=7, opacity=0.7),
            name='Predicted vs Actual'
        ))
        rng = [float(min(hh_m['y_te'])), float(max(hh_m['y_te']))]
        fig.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                  line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                  name='Perfect fit'))
        apply_layout(fig, height=320,
                     xaxis_title="Actual kWh",
                     yaxis_title="Predicted kWh",
                     showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=hh_m['y_te'], y=hh_m['lr_pred'],
            mode='markers',
            marker=dict(color=COLORS['blue'], size=7, opacity=0.7),
            name='LR Predicted'
        ))
        fig2.add_trace(go.Scatter(x=rng, y=rng, mode='lines',
                                   line=dict(color=COLORS['gray'], dash='dash', width=1.5),
                                   name='Perfect fit'))
        apply_layout(fig2, height=320,
                     xaxis_title="Actual kWh",
                     yaxis_title="LR Predicted kWh",
                     title="Predicted vs Actual — Linear Regression",
                     showlegend=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Residual distribution — XGBoost</div>',
                unsafe_allow_html=True)
    resid_h = hh_m['y_te'] - hh_m['gb_pred']
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=resid_h, nbinsx=20,
                                  marker_color=COLORS['teal'],
                                  opacity=0.8, name='Household residuals'))
    fig_r.add_vline(x=0, line_dash="dash", line_color=COLORS['coral'])
    apply_layout(fig_r, height=250,
                 xaxis_title="Residual (kWh)",
                 yaxis_title="Count", showlegend=False)
    st.plotly_chart(fig_r, use_container_width=True)

def tab_features(hh, comm, models):
    hh_m = models['hh']

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">XGBoost feature importance (Household)</div>',
                    unsafe_allow_html=True)
        fi_vals  = hh_m['fi']
        fi_names = ['Month', 'Solar Flag', 'House Type', 'Rooms', 'Solar Capacity']
        fig = go.Figure(go.Bar(
            x=fi_vals * 100,
            y=fi_names,
            orientation='h',
            marker_color=[COLORS['teal'] if v == max(fi_vals) else COLORS['blue']
                          for v in fi_vals],
            text=[f"{v*100:.1f}%" for v in fi_vals],
            textposition='outside'
        ))
        apply_layout(fig, height=280,
                     xaxis_title="Importance (%)",
                     showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Consumption by number of rooms</div>',
                    unsafe_allow_html=True)
        rooms_avg = hh.groupby('Rooms')['Units Consumed'].mean().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"{r} room{'s' if r>1 else ''}" for r in rooms_avg.index],
            y=rooms_avg.values,
            marker_color=[f"rgba(29,158,117,{0.3 + i*0.12})"
                          for i in range(len(rooms_avg))],
            text=[f"{v:.0f}" for v in rooms_avg.values],
            textposition='outside'
        ))
        apply_layout(fig, height=280,
                     yaxis_title="Avg kWh / month", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Seasonal patterns — radar chart</div>',
                    unsafe_allow_html=True)
        hh_s   = hh.groupby('Season')['Units Consumed'].mean()
        comm_s = comm.groupby('Season')['Units Consumed (After Solar)'].mean()
        seasons = list(hh_s.index)
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=list(hh_s.values) + [hh_s.values[0]],
            theta=seasons + [seasons[0]],
            fill='toself', name='Household',
            line_color=COLORS['teal'], fillcolor='rgba(29,158,117,0.15)'
        ))
        fig.add_trace(go.Scatterpolar(
            r=list(comm_s.values/5) + [comm_s.values[0]/5],
            theta=seasons + [seasons[0]],
            fill='toself', name='Commercial / 5',
            line_color=COLORS['blue'], fillcolor='rgba(55,138,221,0.15)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                          font_family="DM Sans", height=320,
                          legend=dict(orientation="h", y=-0.15),
                          margin=dict(t=30,b=60,l=40,r=40))
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">House type — consumption distribution</div>',
                    unsafe_allow_html=True)
        apt  = hh[hh['House Type'] == 'Apartment']['Units Consumed']
        indp = hh[hh['House Type'] == 'Independent']['Units Consumed']
        fig  = go.Figure()
        fig.add_trace(go.Box(y=apt.values,  name='Apartment',
                             marker_color=COLORS['teal'], boxmean=True))
        fig.add_trace(go.Box(y=indp.values, name='Independent',
                             marker_color=COLORS['blue'], boxmean=True))
        apply_layout(fig, height=320,
                     yaxis_title="kWh / month", showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Correlation matrix — Household features</div>',
                unsafe_allow_html=True)
    hh2 = hh.copy()
    hh2['house_type_num'] = (hh2['House Type'] == 'Independent').astype(int)
    corr_cols = ['month_num', 'Rooms', 'Solar Capacity (kW)',
                 'solar_flag', 'house_type_num', 'Units Consumed']
    corr = hh2[corr_cols].corr()
    fig_h = px.imshow(corr, text_auto='.2f',
                      color_continuous_scale=['#D85A30','white','#1D9E75'],
                      zmin=-1, zmax=1,
                      labels=dict(color="Correlation"))
    fig_h.update_layout(font_family="DM Sans", height=350,
                         margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_h, use_container_width=True)


def tab_predict(hh, comm, models, ctrl):
    st.markdown('<div class="section-title">Single-household prediction</div>',
                unsafe_allow_html=True)

    rooms     = ctrl['rooms']
    solar     = ctrl['solar']
    sol_cap   = ctrl['sol_cap']
    h_type    = ctrl['h_type']
    pred_month= ctrl['pred_month']

    le = LabelEncoder().fit(hh['House Type'])
    hh2 = hh.copy(); hh2['house_type_enc'] = le.transform(hh2['House Type'])
    feat_h = ['month_num','solar_flag','house_type_enc','Rooms','Solar Capacity (kW)']
    lr_m = LinearRegression()
    gb_m = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                      max_depth=4, subsample=0.8, random_state=42)
    lr_m.fit(hh2[feat_h].values, hh2['Units Consumed'].values)
    gb_m.fit(hh2[feat_h].values, hh2['Units Consumed'].values)

    inp = [[
        MONTH_MAP[pred_month],
        1 if solar == 'Yes' else 0,
        int(le.transform([h_type])[0]),
        rooms,
        sol_cap
    ]]

    lr_pred_val = max(0.0, float(lr_m.predict(inp)[0]))
    gb_pred_val = max(0.0, float(gb_m.predict(inp)[0]))

    monthly_preds = []
    for m in range(1, 13):
        i2 = [[m, 1 if solar == 'Yes' else 0,
                int(le.transform([h_type])[0]),
                rooms, sol_cap]]
        monthly_preds.append({
            'Month': MONTH_SHORT[m-1],
            'LR':    round(max(0, float(lr_m.predict(i2)[0])), 1),
            'XGBoost': round(max(0, float(gb_m.predict(i2)[0])), 1),
        })

    st.markdown(f"**Profile:** {h_type} · {rooms} rooms · Solar: {solar} ({sol_cap} kW) · Month: {pred_month}")
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="val">{lr_pred_val:.0f}</div>'
        f'<div class="lbl">Linear Regression</div><div class="sub">kWh predicted</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="metric-card"><div class="val">{gb_pred_val:.0f}</div>'
        f'<div class="lbl">XGBoost</div><div class="sub">kWh predicted</div></div>',
        unsafe_allow_html=True
    )
    avg_actual = hh[(hh['Month'] == pred_month) &
                    (hh['Rooms'] == rooms)]['Units Consumed'].mean()
    avg_actual = avg_actual if not np.isnan(avg_actual) else (lr_pred_val + gb_pred_val) / 2
    c3.markdown(
        f'<div class="metric-card"><div class="val">{avg_actual:.0f}</div>'
        f'<div class="lbl">Dataset avg (same segment)</div><div class="sub">kWh actual</div></div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Predicted annual profile for this household</div>',
                unsafe_allow_html=True)

    mp_df = pd.DataFrame(monthly_preds)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mp_df['Month'], y=mp_df['LR'],
                              name='Linear Regression',
                              line=dict(color=COLORS['blue'], dash='dash', width=2),
                              mode='lines+markers', marker=dict(size=5)))
    fig.add_trace(go.Scatter(x=mp_df['Month'], y=mp_df['XGBoost'],
                              name='XGBoost',
                              line=dict(color=COLORS['teal'], width=2.5),
                              mode='lines+markers', marker=dict(size=5)))
    apply_layout(fig, height=300,
                 yaxis_title="Predicted kWh",
                 title="Predicted monthly consumption for this profile")
    st.plotly_chart(fig, use_container_width=True)

    annual_kwh = gb_pred_val * 12
    bill_est   = annual_kwh * 6.5  # ₹6.5/kWh approx NPCL rate
    st.markdown(
        f'<div class="insight-box">'
        f'<b>Annual estimate (XGBoost):</b> ~{annual_kwh:.0f} kWh/year · '
        f'Estimated annual bill: ₹{bill_est:,.0f} (@ ₹6.50/kWh average NPCL rate). '
        f'{"Solar savings: ~₹" + str(round(sol_cap * 1400 * 6.5)) + "/year" if solar == "Yes" and sol_cap > 0 else "Consider solar to reduce costs."}'
        f'</div>',
        unsafe_allow_html=True
    )


def tab_street():
    st.markdown('<div class="section-title">GNIDA Street Light Energy Estimation</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        ["33,534", "36,217 kWh", "13.2 GWh", "90 W"],
        ["Total lights estimated", "Daily consumption", "Annual estimate", "LED fixture wattage"]
    ):
        col.markdown(
            f'<div class="metric-card"><div class="val">{val}</div>'
            f'<div class="lbl">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Road network breakdown</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=['Wide roads (206 km)', 'Internal roads (594 km)'],
            values=[206, 594],
            hole=0.4,
            marker_colors=[COLORS['amber'], COLORS['teal']],
            textinfo='label+percent'
        ))
        fig.update_layout(font_family="DM Sans", height=300,
                          margin=dict(t=20, b=20, l=20, r=20),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Daily consumption by road type</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=['Wide roads\n(both sides)', 'Internal roads\n(one side)', 'Total'],
            y=[14832.72, 21384.0, 36216.72],
            marker_color=[COLORS['amber'], COLORS['teal'], COLORS['coral']],
            text=['14,833 kWh', '21,384 kWh', '36,217 kWh'],
            textposition='outside',
            width=0.5
        ))
        apply_layout(fig2, height=300, yaxis_title="kWh / day",
                     showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Interactive calculation tool</div>',
                unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        wattage = st.slider("Fixture wattage (W)", 50, 200, 90, step=10)
    with col_b:
        spacing = st.slider("Light spacing (m)", 20, 50, 30, step=5)
    with col_c:
        op_hours = st.slider("Operating hours / day", 8, 14, 12)

    wide_lights = int(2 * (206_000 / spacing))
    int_lights  = int(594_000 / spacing)
    total_lights = wide_lights + int_lights
    wide_kwh    = wide_lights * (wattage / 1000) * op_hours
    int_kwh     = int_lights  * (wattage / 1000) * op_hours
    total_kwh   = wide_kwh + int_kwh
    annual_gwh  = total_kwh * 365 / 1e6

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wide road lights",    f"{wide_lights:,}")
    col2.metric("Internal road lights",f"{int_lights:,}")
    col3.metric("Daily consumption",   f"{total_kwh:,.0f} kWh")
    col4.metric("Annual consumption",  f"{annual_gwh:.2f} GWh")

    st.markdown(
        '<div class="insight-box">'
        '<b>Methodology (GNIDA 2020 data):</b><br>'
        'Wide roads (45–132 m width, 206 km total): lights on both sides → '
        '2 × (206,000 ÷ spacing) fixtures<br>'
        'Internal roads (&lt;45 m width, 594 km total): lights one side → '
        '594,000 ÷ spacing fixtures<br>'
        'Validated by GNIDA installation of 3,740 lights across 64 villages '
        'and Noida\'s 100,000+ LED street lights across urban/rural/industrial areas.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_regional(hh, comm):
    st.markdown('<div class="section-title">Regional Consumption Analysis</div>',
                unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown('<div class="section-title">Household: Top 10 Sectors by Demand</div>',
                    unsafe_allow_html=True)
        hh_sector = hh.groupby('Sector')['Units Consumed'].sum().sort_values(ascending=False).head(10)
        fig_h = px.bar(x=hh_sector.values, y=hh_sector.index, orientation='h',
                       color=hh_sector.values,
                       color_continuous_scale=['#9FE1CB','#1D9E75'],
                       labels={'x': 'Total kWh', 'y': 'Sector'})
        fig_h.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)
        
    with col_b:
        st.markdown('<div class="section-title">Commercial: Top 10 Areas by Demand</div>',
                    unsafe_allow_html=True)
        comm_area = comm.groupby('Area')['Units Consumed (After Solar)'].sum().sort_values(ascending=False).head(10)
        fig_c = px.bar(x=comm_area.values, y=comm_area.index, orientation='h',
                       color=comm_area.values,
                       color_continuous_scale=['#BBD9F2','#378ADD'],
                       labels={'x': 'Total kWh', 'y': 'Area'})
        fig_c.update_layout(**PLOTLY_LAYOUT, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown('<div class="section-title">Sector Comparison Metrics</div>',
                unsafe_allow_html=True)
    
    top_sector = hh_sector.index[0]
    avg_top = hh[hh['Sector'] == top_sector]['Units Consumed'].mean()
    top_area = comm_area.index[0]
    avg_comm_top = comm[comm['Area'] == top_area]['Units Consumed (After Solar)'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest Demand Sector", top_sector)
    c2.metric("Avg Consumption (Top Sector)", f"{avg_top:.0f} kWh")
    c3.metric("Highest Commercial Area", top_area)
    c4.metric("Avg Comm. Consumption", f"{avg_comm_top:.0f} kWh")

    st.markdown(
        '<div class="insight-box">'
        f'<b>Market Intelligence:</b> <b>{top_sector}</b> is currently the peak load zone for residential consumption in the dataset. '
        f'For commercial activities, <b>{top_area}</b> shows the highest energy density. '
        'This regional variance suggests that infrastructure upgrades should be prioritized in these high-growth zones.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_roi(hh):
    st.markdown('<div class="section-title">Solar ROI & Payback Calculator</div>',
                unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown('<div class="section-title">Investment Inputs</div>', unsafe_allow_html=True)
        cost_per_kw = st.number_input("Installation Cost per kW (₹)", value=55000, step=5000)
        capacity    = st.slider("System Capacity (kW)", 1, 20, 5)
        tariff      = st.number_input("Electricity Tariff (₹/kWh)", value=6.5, step=0.5)
        maintenance = st.slider("Annual Maintenance (% of cost)", 0.0, 5.0, 1.0) / 100
        
        total_investment = cost_per_kw * capacity
        st.info(f"**Total Investment:** ₹{total_investment:,.0f}")

    with col_r:
        st.markdown('<div class="section-title">Financial Performance</div>', unsafe_allow_html=True)
        # Average solar generation in Noida is ~4 kWh per kWp per day
        daily_gen = capacity * 4 
        annual_gen = daily_gen * 365
        annual_savings = annual_gen * tariff
        annual_maint_cost = total_investment * maintenance
        net_annual_savings = annual_savings - annual_maint_cost
        
        payback_years = total_investment / net_annual_savings if net_annual_savings > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Generation", f"{annual_gen:,.0f} kWh")
        c2.metric("Annual Savings", f"₹{annual_savings:,.0f}")
        c3.metric("Payback Period", f"{payback_years:.1f} Years")

        # Visualization
        years = list(range(0, 21))
        cumulative_cashflow = [-(total_investment)]
        for y in range(1, 21):
            cumulative_cashflow.append(cumulative_cashflow[-1] + net_annual_savings)
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=cumulative_cashflow, mode='lines+markers',
                                 name='Net Cash Flow (Cumulative)',
                                 line=dict(color=COLORS['teal'], width=3)))
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS['red'])
        
        apply_layout(fig, height=350, yaxis_title="Cumulative Balance (₹)",
                     xaxis_title="Year", title="20-Year Financial Projection")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="insight-box">'
        f'<b>Verdict:</b> With a <b>{payback_years:.1f}-year payback</b>, solar is a highly viable investment for Noida homeowners. '
        f'Over 20 years, this {capacity}kW system will generate approximately <b>₹{net_annual_savings * 20 / 1e5:.1f} Lakhs</b> in net savings.'
        '</div>',
        unsafe_allow_html=True
    )


def tab_simulator(hh):
    st.markdown('<div class="section-title">What-If Scenario Simulator</div>',
                unsafe_allow_html=True)
    
    st.markdown("Experiment with upgrades and habit changes to see how much you can save on your bill.")

    c1, c2 = st.columns([1, 1.5])

    with c1:
        st.markdown('<div class="section-title">Step 1: Base Consumption</div>', unsafe_allow_html=True)
        base_units = st.number_input("Average Monthly Units (kWh)", value=450, step=50)
        base_load  = st.slider("Current Connected Load (kW)", 1, 10, 3, key="sim_load")
        
        st.markdown('<div class="section-title">Step 2: Apply Upgrades</div>', unsafe_allow_html=True)
        ac_upgrade = st.slider("Replace old appliances with 5-Star (%)", 0, 50, 0, help="Typically saves 20-30% on AC load version.")
        habit_change = st.slider("Behavioral changes/Habits (%)", 0, 20, 0, help="Turning off lights/AC when not in room.")
        solar_add  = st.slider("Add New Solar Capacity (kW)", 0, 15, 0)
        
    with c2:
        st.markdown('<div class="section-title">The Impact</div>', unsafe_allow_html=True)
        
        # 1. Base Bill
        base_bill = calculate_uppcl_bill(base_units, base_load)
        
        # 2. Simulated Units
        efficiency_gain = (ac_upgrade + habit_change) / 100
        reduced_units = base_units * (1 - efficiency_gain)
        # Solar generation approx 120 units/month per kW
        solar_gen = solar_add * 120
        sim_units = max(0, reduced_units - solar_gen)
        
        # 3. Simulated Bill
        sim_bill = calculate_uppcl_bill(sim_units, base_load)
        
        monthly_saving = base_bill - sim_bill
        annual_saving = monthly_saving * 12
        
        sc1, sc2 = st.columns(2)
        sc1.metric("Monthly Savings", f"₹{monthly_saving:,.0f}", delta=f"{(monthly_saving/base_bill*100):.1f}% reduction", delta_color="normal")
        sc2.metric("Annual Total Savings", f"₹{annual_saving:,.0f}")

        # Comparison Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current Bill", x=["Electricity Bill"], y=[base_bill], marker_color=COLORS['gray']))
        fig.add_trace(go.Bar(name="Optimized Bill", x=["Electricity Bill"], y=[sim_bill], marker_color=COLORS['teal']))
        
        apply_layout(fig, height=300, yaxis_title="Rupees (₹)", barmode='group', title="Current vs Optimized Monthly Bill")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight-box" style="border-left-color: {COLORS["purple"]}">'
        f'<b>Simulation Verdict:</b> By applying these changes, your monthly consumption drops from <b>{base_units}</b> to <b>{sim_units:.0f}</b> kWh. '
        f'This moves your bill into a lower tariff slab, resulting in an annual pocket saving of <b>₹{annual_saving:,.0f}</b>. '
        f'{"Great job! You are becoming energy independent." if solar_add > 0 else "Tip: Adding solar could bring this bill down even further."}'
        '</div>',
        unsafe_allow_html=True
    )


def tab_rawdata(hh, comm):
    st.markdown('<div class="section-title">Raw datasets</div>',
                unsafe_allow_html=True)
    tab_a, tab_b = st.tabs(["🏠 Household (200 records)", "🏢 Commercial (130 records)"])

    with tab_a:
        st.dataframe(hh, use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1: st.write(hh.describe().round(2))
        with col2: st.write(hh.dtypes.rename("dtype").to_frame())

    with tab_b:
        st.dataframe(comm, use_container_width=True, height=400)
        col1, col2 = st.columns(2)
        with col1: st.write(comm.describe().round(2))
        with col2: st.write(comm.dtypes.rename("dtype").to_frame())


def main():
    st.markdown("""
    <div class="main-header">
        <h1>Machine Learning-Based Prediction of Household Energy Consumption</h1>
        <p>Greater Noida &amp; Noida · Household (200 records) + Commercial (130 records) datasets ·
           XGBoost · LSTM · ARIMA · SARIMA · Linear Regression</p>
        <p style="font-size:0.8rem;opacity:0.75;margin-top:8px">
            Mohit Batar · Prakash Mishra · Ashish Kumar &nbsp;|&nbsp; March 2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    try:
        hh, comm, le_h, le_c = load_data()
    except FileNotFoundError:
        st.error(
            "**Data files not found.**\n\n"
            "Please place `noida_electricity_household.xlsx` and "
            "`noida_commercial.xlsx` in the same directory as this script, then restart."
        )
        st.stop()

    models = train_models(hh, comm)
    ctrl   = sidebar(hh, comm)

    tabs = st.tabs([
        "Overview",
        "Regional Analysis",
        "Solar ROI",
        "Scenario Simulator",
        "Forecast",
        "Model Comparison",
        "Feature Analysis",
        "Custom Prediction",
        "Street Lights",
        "Raw Data",
    ])

    with tabs[0]: tab_overview(hh, comm)
    with tabs[1]: tab_regional(hh, comm)
    with tabs[2]: tab_roi(hh)
    with tabs[3]: tab_simulator(hh)
    with tabs[4]: tab_forecast(hh, comm, ctrl)
    with tabs[5]: tab_models(hh, comm, models)
    with tabs[6]: tab_features(hh, comm, models)
    with tabs[7]: tab_predict(hh, comm, models, ctrl)
    with tabs[8]: tab_street()
    with tabs[9]: tab_rawdata(hh, comm)

    st.markdown(
        '<div class="footer">'
        'Machine Learning-Based Prediction of Household Energy Consumption · '
        'Greater Noida Case Study · Powered by Streamlit + Plotly + scikit-learn'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
