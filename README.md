# Noida Energy Analytics

A Streamlit application for analyzing and forecasting energy consumption (Household and Commercial) in Noida.

## Features
- **Dataset Summary**: Descriptive statistics of electricity consumption.
- **Forecasting Models**: ARIMA, SARIMA, LSTM, Linear Regression, and XGBoost.
- **Feature analysis**: Impact of room count, solar panels, and seasonality.
- **Interactive Predictor**: Predict future consumption based on custom user profiles.

## Technologies Used
- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Machine Learning**: Scikit-Learn, NumPy, Pandas
- **Excel Processing**: Openpyxl

## How to Run Locally

### 1. Install Python
Ensure you have Python 3.8+ installed.

### 2. Install Dependencies
Open your terminal in this folder and run:
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run energy_consumption_app.py
```

## How to Push to GitHub

1. **Initialize Git**:
   ```bash
   git init
   ```
2. **Add Files**:
   ```bash
   git add .
   ```
3. **Commit**:
   ```bash
   git commit -m "Initial commit: Noida Energy Analytics App"
   ```
4. **Create a Repository on GitHub**:
   - Go to [GitHub](https://github.com/new).
   - Create a new repository (do not initialize with README).
   - Copy the HTTPS/SSH URL.

5. **Connect and Push**:
   ```bash
   git remote add origin <YOUR_GITHUB_REPO_URL>
   git branch -M main
   git push -u origin main
   ```
