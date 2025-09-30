# 🌍 Air Quality Forecasting System

Machine learning-based system for predicting air quality index (AQI) and pollutant levels.

## Features
- Time series forecasting (ARIMA, Prophet, LSTM, XGBoost)
- Interactive dashboard with visualizations
- AQI alerts and warnings
- Admin panel for data management

## Setup
See SETUP.md for detailed installation instructions

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `python scripts/download_data.py`
3. Train models: `python scripts/train_models.py`
4. Run dashboard: `streamlit run dashboard/app.py`

## Project Structure
- `src/`: Source code
- `data/`: Datasets
- `models/`: Trained models
- `dashboard/`: Streamlit app
- `configs/`: Configuration files