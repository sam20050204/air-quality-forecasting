import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.api_calculator import AQICalculator
from src.models.xgboost_model import XGBoostModel
from src.data.data_uploader import CSVUploader
from src.data.prediction_handler import PredictionHandler

# Page configuration
st.set_page_config(
    page_title=config.app_config['app']['title'],
    page_icon=config.app_config['app']['page_icon'],
    layout=config.app_config['app']['layout']
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'prediction_results' not in st.session_state:
    st.session_state.prediction_results = None

@st.cache_data
def load_sample_data():
    """Load sample air quality data"""
    data_path = Path('data/raw/delhi_air_quality.csv')
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        return df
    else:
        # Generate sample data if file doesn't exist
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
        df = pd.DataFrame({
            'PM2.5': np.random.uniform(30, 150, len(dates)),
            'PM10': np.random.uniform(50, 250, len(dates)),
            'NO2': np.random.uniform(10, 80, len(dates)),
            'SO2': np.random.uniform(5, 40, len(dates)),
            'CO': np.random.uniform(0.5, 3, len(dates)),
            'O3': np.random.uniform(20, 100, len(dates)),
        }, index=dates)
        df.index.name = 'datetime'
        return df

def load_models():
    """Load trained models"""
    models = {}
    model_dir = Path('models/saved_models')
    
    if not model_dir.exists():
        return None
    
    for pollutant in config.data_config['pollutants']:
        model_path = model_dir / f'xgboost_{pollutant.lower()}.pkl'
        if model_path.exists():
            try:
                model = XGBoostModel()
                model.load_model(str(model_path))
                models[pollutant] = model
            except Exception as e:
                st.warning(f"Could not load model for {pollutant}: {str(e)}")
    
    return models if models else None

def calculate_aqi_from_pollutants(row):
    """Calculate overall AQI from all pollutants"""
    pm25_aqi = AQICalculator.calculate_aqi('PM2.5', row.get('PM2.5', 0))
    return pm25_aqi if pm25_aqi else 0

def create_gauge_chart(value, title, max_value=500):
    """Create a gauge chart for AQI"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 50], 'color': '#00e400'},
                {'range': [50, 100], 'color': '#ffff00'},
                {'range': [100, 150], 'color': '#ff7e00'},
                {'range': [150, 200], 'color': '#ff0000'},
                {'range': [200, 300], 'color': '#8f3f97'},
                {'range': [300, 500], 'color': '#7e0023'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            }
        }
    ))
    
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=60, b=20))
    return fig

def plot_time_series(df, pollutant):
    """Plot time series for a pollutant"""
    fig = px.line(df, x=df.index, y=pollutant, title=f'{pollutant} Levels Over Time')
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title=f'{pollutant} (μg/m³)',
        hovermode='x unified'
    )
    return fig

def plot_prediction_comparison(result_df, pollutant):
    """Plot actual vs predicted values"""
    fig = go.Figure()
    
    actual_col = f'{pollutant}_actual'
    predicted_col = f'{pollutant}_predicted'
    
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df[actual_col],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=result_df.index,
        y=result_df[predicted_col],
        mode='lines',
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{pollutant} - Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title=f'{pollutant} (μg/m³)',
        hovermode='x unified',
        height=400
    )
    
    return fig

def show_dashboard(df):
    """Main dashboard view"""
    st.header("📊 Current Air Quality Dashboard")
    
    # Get latest data
    latest = df.iloc[-1]
    
    # Calculate current AQI
    current_aqi = calculate_aqi_from_pollutants(latest)
    category, color = AQICalculator.get_aqi_category(current_aqi)
    
    # Display current AQI
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.plotly_chart(create_gauge_chart(current_aqi, "Current AQI"), use_container_width=True)
    
    with col2:
        st.metric("AQI Category", category)
        st.metric("Location", "Delhi")
        st.metric("Last Updated", latest.name.strftime('%Y-%m-%d %H:%M'))
    
    with col3:
        st.metric("PM2.5", f"{latest['PM2.5']:.1f} μg/m³")
        st.metric("PM10", f"{latest['PM10']:.1f} μg/m³")
        st.metric("NO2", f"{latest['NO2']:.1f} μg/m³")
    
    # Health recommendations
    st.subheader("💡 Health Recommendations")
    if current_aqi <= 50:
        st.success("Air quality is good. Enjoy outdoor activities!")
    elif current_aqi <= 100:
        st.info("Air quality is moderate. Sensitive individuals should consider reducing prolonged outdoor exertion.")
    elif current_aqi <= 150:
        st.warning("Unhealthy for sensitive groups. Reduce prolonged or heavy outdoor exertion.")
    else:
        st.error("Air quality is unhealthy. Avoid outdoor activities and stay indoors.")
    
    # Recent trends
    st.subheader("📈 Recent Trends (Last 7 Days)")
    
    recent_df = df.tail(168)  # Last 7 days (hourly data)
    
    pollutant_select = st.selectbox("Select Pollutant", config.data_config['pollutants'])
    st.plotly_chart(plot_time_series(recent_df, pollutant_select), use_container_width=True)

def show_upload_page():
    """CSV Upload and Prediction Page"""
    st.header("📤 Upload & Predict")
    
    st.markdown("""
    Upload your air quality CSV file to get predictions and insights.
    
    **Required Format:**
    - Must include a datetime column
    - Should contain at least one pollutant: PM2.5, PM10, NO2, SO2, CO, or O3
    - Minimum 50 rows of data
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload your air quality data in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_uploaded = pd.read_csv(uploaded_file)
            
            # Initialize uploader
            uploader = CSVUploader()
            
            # Display raw data preview
            st.subheader("📋 Data Preview")
            st.dataframe(df_uploaded.head(10), use_container_width=True)
            
            # Validate CSV
            is_valid, errors = uploader.validate_csv(df_uploaded)
            
            if not is_valid:
                st.error("❌ Validation Failed")
                for error in errors:
                    st.write(f"- {error}")
                return
            
            st.success("✅ CSV file validated successfully!")
            
            # Preprocess data
            with st.spinner("Preprocessing data..."):
                df_processed = uploader.preprocess_uploaded_data(df_uploaded)
                st.session_state.uploaded_data = df_processed
            
            # Show data summary
            st.subheader("📊 Data Summary")
            summary = uploader.get_data_summary(df_processed)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", summary['total_rows'])
            with col2:
                st.metric("Date Range", f"{summary['date_range']['start']} to {summary['date_range']['end']}")
            with col3:
                st.metric("Data Quality", summary['data_quality'])
            
            st.info("💡 Note: Models must be trained first. Run: `python scripts/train_models.py --pollutant all`")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    else:
        # Show example CSV format
        st.subheader("📝 Example CSV Format")
        
        example_data = {
            'datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
            'PM2.5': [45.2, 52.1, 48.9],
            'PM10': [78.5, 85.2, 81.3],
            'NO2': [32.1, 35.6, 33.8],
            'SO2': [12.5, 14.2, 13.1],
            'CO': [1.2, 1.4, 1.3],
            'O3': [45.2, 48.1, 46.5]
        }
        
        st.dataframe(pd.DataFrame(example_data), use_container_width=True)
        
        st.info("💡 Your CSV should have similar columns. Datetime column is required, pollutants are optional.")

def show_forecasts(df):
    """Forecasts page"""
    st.header("🔮 Air Quality Forecasts")
    
    st.info("💡 Note: Models must be trained first. Run: `python scripts/train_models.py --pollutant all`")
    
    st.write("This page will show forecasts once models are trained.")

def show_analysis(df):
    """Data analysis page"""
    st.header("📈 Data Analysis")
    
    # Data overview
    st.subheader("📊 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Time Span", f"{(df.index[-1] - df.index[0]).days} days")
    col3.metric("Pollutants", len([col for col in df.columns if col in config.data_config['pollutants']]))
    col4.metric("Completeness", f"{((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100):.1f}%")
    
    # Pollutant statistics
    st.subheader("🔬 Pollutant Statistics")
    
    pollutants_in_data = [col for col in df.columns if col in config.data_config['pollutants']]
    
    if pollutants_in_data:
        stats_df = df[pollutants_in_data].describe().T
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)

def show_about():
    """About page"""
    st.header("ℹ️ About Air Quality Forecasting System")
    
    st.markdown("""
    ## 🌍 Overview
    
    This Air Quality Forecasting System uses advanced machine learning models to predict 
    pollutant levels and provide actionable insights for better air quality management.
    
    ## 🎯 Features
    
    - **Real-time Monitoring**: Track current air quality levels
    - **Advanced Forecasting**: Predict future pollutant levels
    - **Data Upload & Analysis**: Upload your own data for predictions
    - **Comprehensive Analytics**: Statistical analysis and insights
    
    ## 🚀 Getting Started
    
    1. **Train Models** (if not done):
       ```bash
       python scripts/train_models.py --pollutant all
       ```
    
    2. Explore the dashboard features
    3. Upload your own data for analysis
    
    ## 📞 Support
    
    Check the README.md file for detailed documentation.
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: October 2025
    """)

def main():
    """Main app function"""
    # Header
    st.markdown('<h1 class="main-header">🌍 Air Quality Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", [
        "Dashboard", 
        "Upload & Predict", 
        "Forecasts", 
        "Data Analysis", 
        "About"
    ])
    
    # Load data
    try:
        df = load_sample_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Run: `python quick_fix.py` to generate sample data")
        return
    
    # Route to appropriate page
    if page == "Dashboard":
        show_dashboard(df)
    elif page == "Upload & Predict":
        show_upload_page()
    elif page == "Forecasts":
        show_forecasts(df)
    elif page == "Data Analysis":
        show_analysis(df)
    elif page == "About":
        show_about()

# THIS IS THE CRITICAL FIX - Call main() when script runs
if __name__ == "__main__":
    main()