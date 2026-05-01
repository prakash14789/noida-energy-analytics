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
from src.config import MONTH_MAP, MONTH_SHORT, COLORS_THEME, COLORS, apply_layout, MONTH_ORDER
from src.data_processing import arima_forecast, sarima_forecast, lstm_forecast, metrics, calculate_uppcl_bill

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
        apply_layout(fig, height=280,
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
        apply_layout(fig_h, height=400, coloraxis_showscale=False)
        st.plotly_chart(fig_h, use_container_width=True)
        
    with col_b:
        st.markdown('<div class="section-title">Commercial: Top 10 Areas by Demand</div>',
                    unsafe_allow_html=True)
        comm_area = comm.groupby('Area')['Units Consumed (After Solar)'].sum().sort_values(ascending=False).head(10)
        fig_c = px.bar(x=comm_area.values, y=comm_area.index, orientation='h',
                       color=comm_area.values,
                       color_continuous_scale=['#BBD9F2','#378ADD'],
                       labels={'x': 'Total kWh', 'y': 'Area'})
        apply_layout(fig_c, height=400, coloraxis_showscale=False)
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

def tab_3d_map(hh, comm, models):
    import pydeck as pdk
    import os
    import json
    
    @st.cache_data
    def load_osm_data(filename):
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except: return None
        return None

    @st.cache_data
    def get_building_data_linked(_buildings_raw, _hh, _comm, _models, _loc_map, mode, month_num=5):
        if not _buildings_raw: return []
        import numpy as np
        
        sectors = []
        for name, coords in _loc_map.items():
            sectors.append({'name': name, 'lat': coords[0], 'lon': coords[1]})
        
        hh_hist = _hh.groupby('Sector')['Units Consumed'].mean().to_dict()
        comm_hist = _comm.groupby('Area')['Units Consumed (After Solar)'].mean().to_dict()
        
        model = _models['hh']['gb']
        sector_preds = {}
        for s in sectors:
            inp = [[month_num, 0.2, 0, 3, 0]]
            pred = model.predict(inp)[0]
            sector_preds[s['name']] = float(pred)
            
        # Prepare for normalization
        all_vals = list(sector_preds.values()) + list(hh_hist.values()) + list(comm_hist.values())
        min_v = min(all_vals) if all_vals else 0
        max_v = max(all_vals) if all_vals else 1000
        
        def get_gradient_color(v):
            norm = (v - min_v) / (max_v - min_v + 1e-6)
            norm = max(0, min(1, norm))
            if norm < 0.5:
                r = int(255 * (norm * 2))
                g = 255
                b = 0
            else:
                r = 255
                g = int(255 * (1 - (norm - 0.5) * 2))
                b = 0
            return [r, g, b, 200]

        building_list = []
        for feature in _buildings_raw['features']:
            try:
                geom_type = feature['geometry']['type']
                coords = feature['geometry']['coordinates']
                
                if geom_type == 'Polygon':
                    poly_coords = coords[0]
                    pts = np.array(poly_coords)
                    b_lon, b_lat = np.mean(pts, axis=0)
                elif geom_type == 'MultiPolygon':
                    poly_coords = coords[0][0]
                    pts = np.array(poly_coords)
                    b_lon, b_lat = np.mean(pts, axis=0)
                else: continue
                
                min_dist = float('inf')
                sector_name = None
                for s in sectors:
                    d = (b_lat - s['lat'])**2 + (b_lon - s['lon'])**2
                    if d < min_dist:
                        min_dist = d
                        sector_name = s['name']
                
                if mode == "ML Predictions":
                    val = sector_preds.get(sector_name, min_v)
                    v_type = "Prediction"
                else:
                    val = hh_hist.get(sector_name, comm_hist.get(sector_name, min_v))
                    v_type = "Historical"
                
                norm_val = (val - min_v) / (max_v - min_v + 1e-6)
                
                building_list.append({
                    'polygon': poly_coords,
                    'sector': sector_name,
                    'value': round(val, 1),
                    'value_type': v_type,
                    'elev': norm_val * 400 + 10,
                    'color': get_gradient_color(val),
                    'lon': b_lon,
                    'lat': b_lat
                })
            except:
                continue
            
        return building_list

    @st.cache_data
    def get_road_coordinates(_roads_raw):
        if not _roads_raw: return []
        road_list = []
        for feature in _roads_raw['features']:
            try:
                coords = feature['geometry']['coordinates']
                if feature['geometry']['type'] == 'LineString':
                    road_list.append({'path': coords})
                elif feature['geometry']['type'] == 'MultiLineString':
                    road_list.append({'path': coords[0]})
            except: continue
        return road_list

    buildings_raw = load_osm_data("buildings_gn.geojson")
    roads_raw = load_osm_data("roads_gn.geojson")

    st.markdown('<div class="section-title">Geospatial Energy Heatmap (3D)</div>',
                unsafe_allow_html=True)
    
    # Mapping sectors to more precise coordinates
    loc_map = {
        # Greater Noida
        'Gamma 1': [28.4744, 77.5029],
        'Delta 1': [28.4833, 77.5144],
        'Alpha 1': [28.4680, 77.5020],
        'Beta 1':  [28.4750, 77.5100],
        'Beta 2':  [28.4680, 77.5140],
        'Zeta 1':  [28.4550, 77.5300],
        'Knowledge Park': [28.4540, 77.4890],
        'Knowledge Park II': [28.4500, 77.4950],
        'Knowledge Park III': [28.4580, 77.4850],
        'Omega 1': [28.4480, 77.5250],
        'Phi 3': [28.4550, 77.5150],
        # Noida
        'Sector 137': [28.5147, 77.4000],
        'Sector 18': [28.5670, 77.3210],
        'Sector 50': [28.5630, 77.3700],
        'Sector 62': [28.6180, 77.3590],
        'Sector 75': [28.5680, 77.3910],
        'Sector 16': [28.5780, 77.3150],
        'Sector 63': [28.6250, 77.3820],
        'Sector 132': [28.5080, 77.3720],
        'Sector 150': [28.4450, 77.4450],
    }


    # Data aggregation
    hh_map = hh.groupby('Sector')['Units Consumed'].sum().reset_index()
    hh_map.rename(columns={'Sector': 'Area', 'Units Consumed': 'units'}, inplace=True)
    hh_map['type'] = 'Household'

    comm_map = comm.groupby('Area')['Units Consumed (After Solar)'].sum().reset_index()
    comm_map.rename(columns={'Units Consumed (After Solar)': 'units'}, inplace=True)
    comm_map['type'] = 'Commercial'

    df_map = pd.concat([hh_map, comm_map])
    df_map['lat'] = df_map['Area'].map(lambda x: loc_map.get(x, [0,0])[0])
    df_map['lon'] = df_map['Area'].map(lambda x: loc_map.get(x, [0,0])[1])
    
    # Filter out unmapped areas
    df_map = df_map[df_map['lat'] != 0]

    col_view, col_info = st.columns([3, 1])
    
    with col_info:
        st.markdown("### Map Controls")
        map_mode = st.radio("Analysis Mode", ["Historical Consumption", "ML Predictions"], index=1)
        view_focus = st.radio("Focus View", ["Greater Noida", "Noida", "All"], index=0)
        
        m_num = 5 # Default May
        if map_mode == "ML Predictions":
            sel_month = st.select_slider("Forecast for Month", options=MONTH_ORDER, value="May")
            m_num = MONTH_MAP[sel_month]
        
        buildings_data = get_building_data_linked(buildings_raw, hh, comm, models, loc_map, map_mode, m_num)
        osm_roads_data = get_road_coordinates(roads_raw)
        
        st.markdown("### Layers")
        show_buildings = st.checkbox("Show 3D Building Energy", value=True)
        show_heatmap = st.checkbox("Show Area Heatmap", value=False)
        show_osm_roads = st.checkbox("Show Real Road Network", value=True)
        
        if show_heatmap:
            intensity = st.slider("Heatmap Intensity", 1, 10, 5)
            radius_pixels = st.slider("Heatmap Radius (px)", 10, 100, 40)

    # Set initial view state based on selection
    if view_focus == "Greater Noida":
        initial_lat, initial_lon, initial_zoom = 28.474, 77.507, 13
    elif view_focus == "Noida":
        initial_lat, initial_lon, initial_zoom = 28.57, 77.35, 12
    else:
        initial_lat, initial_lon, initial_zoom = 28.52, 77.42, 11

    view_state = pdk.ViewState(
        latitude=initial_lat,
        longitude=initial_lon,
        zoom=initial_zoom,
        pitch=60,
        bearing=-30
    )

    layers = []
    
    # 1. 3D Building Energy Layer (Primary)
    if show_buildings and buildings_data:
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                data=buildings_data,
                get_polygon="polygon",
                opacity=0.8,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
                get_elevation="elev",
                get_fill_color="color",
                get_line_color="[255, 255, 255]",
                pickable=True,
                auto_highlight=True,
            )
        )


    if show_osm_roads and osm_roads_data:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=osm_roads_data,
                get_path="path",
                opacity=0.4,
                width_min_pixels=1,
                get_color="[100, 150, 255, 100]",
            )
        )

    # 3. Area Heatmap
    if show_heatmap:
        layers.append(
            pdk.Layer(
                "HeatmapLayer",
                data=df_map,
                get_position=["lon", "lat"],
                get_weight="units",
                radiusPixels=radius_pixels,
                intensity=intensity,
                threshold=0.05,
                opacity=0.7,
            )
        )



    tooltip = {
        "html": "<b>Sector:</b> {sector}<br/>"
                "<b>{value_type} Consumption:</b> {value} kWh",
        "style": {"backgroundColor": "#1A1A1A", "color": "white", "borderRadius": "8px", "fontSize": "0.9rem"}
    }

    with col_view:
        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="mapbox://styles/mapbox/dark-v10" if st.get_option("theme.base") == "dark" else "mapbox://styles/mapbox/light-v10"
        ))

    # Color Legend
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 20px; justify-content: center; margin-top: 5px; font-size: 0.85rem; font-weight: 600;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background: #00FF00; border-radius: 3px;"></div> Low Load
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background: #FFFF00; border-radius: 3px;"></div> Moderate Load
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 15px; height: 15px; background: #FF0000; border-radius: 3px;"></div> Critical Load
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="insight-box">'
        '<b>Digital Twin:</b> This 3D model represents real buildings in Greater Noida. '
        'Color intensity and building height now directly represent energy consumption—either '
        '<b>Historical</b> (actual dataset averages) or <b>ML Predictions</b> (forecasted based on model parameters). '
        '🔴 Red buildings indicate high-load hotspots that require infrastructure attention.'
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

