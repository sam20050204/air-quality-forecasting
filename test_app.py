"""
Minimal test version of the dashboard to diagnose issues
Save this as: test_app.py
Run with: streamlit run test_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Basic page config
st.set_page_config(
    page_title="Air Quality Test",
    page_icon="🌍",
    layout="wide"
)

# Test 1: Basic display
st.title("🌍 Air Quality Forecasting System - TEST MODE")

st.success("✅ Streamlit is working!")

# Test 2: Check imports
st.subheader("Checking Dependencies...")

try:
    import plotly.graph_objects as go
    st.write("✅ Plotly imported")
except Exception as e:
    st.error(f"❌ Plotly error: {e}")

try:
    import xgboost
    st.write("✅ XGBoost imported")
except Exception as e:
    st.error(f"❌ XGBoost error: {e}")

try:
    import yaml
    st.write("✅ YAML imported")
except Exception as e:
    st.error(f"❌ YAML error: {e}")

# Test 3: Check project structure
from pathlib import Path

st.subheader("Checking Project Structure...")

paths_to_check = [
    'configs/model_config.yaml',
    'configs/data_config.yaml',
    'configs/app_config.yaml',
    'data/raw',
    'models/saved_models',
    'src/utils/config.py',
    'src/models/xgboost_model.py',
]

for path in paths_to_check:
    if Path(path).exists():
        st.write(f"✅ {path}")
    else:
        st.write(f"❌ {path} - MISSING")

# Test 4: Try to import project modules
st.subheader("Testing Project Imports...")

try:
    from src.utils.config import config
    st.write("✅ Config module imported")
    st.json({
        "pollutants": config.pollutants,
        "app_title": config.app_config['app']['title']
    })
except Exception as e:
    st.error(f"❌ Config import failed: {e}")
    st.code(str(e))

try:
    from src.models.xgboost_model import XGBoostModel
    st.write("✅ XGBoostModel imported")
except Exception as e:
    st.error(f"❌ XGBoostModel import failed: {e}")
    st.code(str(e))

try:
    from src.data.data_preprocessor import DataPreprocessor
    st.write("✅ DataPreprocessor imported")
except Exception as e:
    st.error(f"❌ DataPreprocessor import failed: {e}")
    st.code(str(e))

# Test 5: Generate sample data
st.subheader("Testing Data Generation...")

try:
    dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
    df = pd.DataFrame({
        'datetime': dates,
        'PM2.5': np.random.uniform(30, 150, len(dates)),
        'PM10': np.random.uniform(50, 250, len(dates)),
    })
    
    st.write("✅ Sample data created")
    st.dataframe(df.head())
    
    # Try plotting
    import plotly.express as px
    fig = px.line(df, x='datetime', y='PM2.5', title='Test Chart')
    st.plotly_chart(fig, use_container_width=True)
    st.write("✅ Plotting works")
    
except Exception as e:
    st.error(f"❌ Data/plotting error: {e}")
    st.code(str(e))

# Test 6: Check data files
st.subheader("Checking Data Files...")

data_file = Path('data/raw/delhi_air_quality.csv')
if data_file.exists():
    try:
        df = pd.read_csv(data_file)
        st.write(f"✅ Data file found: {len(df)} rows")
        st.dataframe(df.head())
    except Exception as e:
        st.error(f"❌ Error reading data: {e}")
else:
    st.warning("⚠️ No data file found at data/raw/delhi_air_quality.csv")
    
    if st.button("Generate Sample Data"):
        data_file.parent.mkdir(parents=True, exist_ok=True)
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'city': 'Delhi',
            'PM2.5': np.random.uniform(10, 200, len(dates)),
            'PM10': np.random.uniform(20, 300, len(dates)),
            'NO2': np.random.uniform(5, 100, len(dates)),
            'SO2': np.random.uniform(2, 50, len(dates)),
            'CO': np.random.uniform(0.1, 5, len(dates)),
            'O3': np.random.uniform(10, 150, len(dates)),
        })
        df.to_csv(data_file, index=False)
        st.success(f"✅ Generated {len(df)} records!")
        st.rerun()

# Test 7: Check models
st.subheader("Checking Models...")

models_dir = Path('models/saved_models')
if models_dir.exists():
    pkl_files = list(models_dir.glob('*.pkl'))
    if pkl_files:
        st.write(f"✅ Found {len(pkl_files)} model files:")
        for f in pkl_files:
            st.write(f"  - {f.name}")
    else:
        st.warning("⚠️ No trained models found")
        st.info("Run: python scripts/train_models.py --pollutant all")
else:
    st.error("❌ models/saved_models directory not found")

# Final status
st.divider()
st.subheader("🎯 Next Steps")

st.markdown("""
**If all checks pass:**
1. Stop this test app (Ctrl+C in terminal)
2. Run the main dashboard: `streamlit run dashboard/app.py`

**If checks fail:**
1. Fix the missing dependencies or files shown above
2. Re-run this test to verify fixes
3. Check the terminal for detailed error messages

**Common Issues:**
- Missing dependencies: `pip install -r requirements.txt`
- Missing configs: Ensure all files are uploaded
- Import errors: Check Python path and file locations
- No data: Click "Generate Sample Data" button above
""")