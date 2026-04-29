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
from src.config import MONTH_MAP, COLORS_THEME, COLORS, MONTH_ORDER, MONTH_SHORT, apply_layout

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

