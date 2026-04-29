import streamlit as st
from src.data_processing import load_data, train_models
from src.ui_tabs import sidebar, tab_overview, tab_forecast, tab_models, tab_features, tab_predict, tab_street, tab_regional, tab_roi, tab_simulator, tab_rawdata

st.set_page_config(
    page_title="Noida Energy Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    # Load the CSS config from the old file directly here, or import it.
    # It executes when we import src.config, because we left the st.markdown calls inside the config.py module scope.
    import src.config
    
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
