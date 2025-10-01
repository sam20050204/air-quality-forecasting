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
        return df
    else:
        # Generate sample data if file doesn't exist
        dates = pd.date_range(end=datetime.now(), periods=168, freq='H')
        df = pd.DataFrame({
            'datetime': dates,
            'city': 'Delhi',
            'PM2.5': np.random.uniform(30, 150, len(dates)),
            'PM10': np.random.uniform(50, 250, len(dates)),
            'NO2': np.random.uniform(10, 80, len(dates)),
            'SO2': np.random.uniform(5, 40, len(dates)),
            'CO': np.random.uniform(0.5, 3, len(dates)),
            'O3': np.random.uniform(20, 100, len(dates)),
        })
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
    fig = px.line(df, x='datetime', y=pollutant, title=f'{pollutant} Levels Over Time')
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

# Add these functions to your dashboard/app.py file
# Place them before the main() function

def show_forecasts(df):
    """Forecasts page"""
    st.header("🔮 Air Quality Forecasts")
    
    # Check if models are loaded
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            models = load_models()
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
            else:
                st.error("⚠️ No trained models found. Please train models first.")
                st.info("Run: `python scripts/train_models.py --pollutant all`")
                return
    
    st.success(f"✅ {len(st.session_state.models)} models loaded")
    
    # Forecast settings
    col1, col2 = st.columns(2)
    
    with col1:
        pollutant = st.selectbox(
            "Select Pollutant",
            list(st.session_state.models.keys())
        )
    
    with col2:
        forecast_hours = st.slider("Forecast Hours", 6, 72, 24)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating forecast..."):
            handler = PredictionHandler()
            handler.models = st.session_state.models
            
            try:
                # Generate forecast
                forecast_df = handler.forecast_future(df, pollutant, hours_ahead=forecast_hours)
                
                # Plot forecast
                fig = go.Figure()
                
                # Historical data (last 48 hours)
                historical = df[pollutant].tail(48)
                fig.add_trace(go.Scatter(
                    x=historical.index,
                    y=historical.values,
                    mode='lines',
                    name='Historical',
                    line=dict(color='blue', width=2)
                ))
                
                # Forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df[f'{pollutant}_forecast'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', dash='dash', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f'{pollutant} - {forecast_hours}h Forecast',
                    xaxis_title='Date & Time',
                    yaxis_title=f'{pollutant} (μg/m³)',
                    hovermode='x unified',
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📊 Forecast Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                forecast_values = forecast_df[f'{pollutant}_forecast']
                
                col1.metric("Average", f"{forecast_values.mean():.1f} μg/m³")
                col2.metric("Maximum", f"{forecast_values.max():.1f} μg/m³")
                col3.metric("Minimum", f"{forecast_values.min():.1f} μg/m³")
                col4.metric("Std Dev", f"{forecast_values.std():.1f}")
                
                # Alert if high values predicted
                threshold = historical.quantile(0.9)
                high_values = forecast_df[forecast_df[f'{pollutant}_forecast'] > threshold]
                
                if len(high_values) > 0:
                    st.warning(f"⚠️ High {pollutant} levels predicted in {len(high_values)} time periods!")
                
                # Download forecast
                csv = forecast_df.to_csv()
                st.download_button(
                    label="📥 Download Forecast",
                    data=csv,
                    file_name=f"{pollutant}_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")


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
        stats_df['missing_%'] = (df[pollutants_in_data].isnull().sum() / len(df) * 100).values
        
        st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Pollutant Correlations")
    
    if len(pollutants_in_data) > 1:
        corr = df[pollutants_in_data].corr()
        
        fig = px.imshow(
            corr,
            labels=dict(color="Correlation"),
            x=pollutants_in_data,
            y=pollutants_in_data,
            color_continuous_scale='RdBu_r',
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("📊 Pollutant Distributions")
    
    selected_pollutant = st.selectbox("Select pollutant for distribution", pollutants_in_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            df,
            x=selected_pollutant,
            nbins=50,
            title=f"{selected_pollutant} Distribution"
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = px.box(
            df,
            y=selected_pollutant,
            title=f"{selected_pollutant} Box Plot"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Time patterns
    st.subheader("⏰ Temporal Patterns")
    
    # Hourly pattern
    df_hourly = df.copy()
    df_hourly['hour'] = df_hourly.index.hour
    hourly_avg = df_hourly.groupby('hour')[selected_pollutant].mean()
    
    fig_hourly = px.line(
        x=hourly_avg.index,
        y=hourly_avg.values,
        title=f"{selected_pollutant} - Average by Hour of Day",
        labels={'x': 'Hour', 'y': f'{selected_pollutant} (μg/m³)'}
    )
    fig_hourly.update_traces(mode='lines+markers')
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    # Day of week pattern
    df_dow = df.copy()
    df_dow['day_of_week'] = df_dow.index.dayofweek
    dow_avg = df_dow.groupby('day_of_week')[selected_pollutant].mean()
    
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    fig_dow = px.bar(
        x=day_names,
        y=dow_avg.values,
        title=f"{selected_pollutant} - Average by Day of Week",
        labels={'x': 'Day', 'y': f'{selected_pollutant} (μg/m³)'}
    )
    st.plotly_chart(fig_dow, use_container_width=True)
    
    # Monthly trend
    if (df.index[-1] - df.index[0]).days > 60:
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.to_period('M')
        monthly_avg = df_monthly.groupby('month')[selected_pollutant].mean()
        
        fig_monthly = px.line(
            x=[str(m) for m in monthly_avg.index],
            y=monthly_avg.values,
            title=f"{selected_pollutant} - Monthly Trend",
            labels={'x': 'Month', 'y': f'{selected_pollutant} (μg/m³)'}
        )
        fig_monthly.update_traces(mode='lines+markers')
        st.plotly_chart(fig_monthly, use_container_width=True)


def show_about():
    """About page"""
    st.header("ℹ️ About Air Quality Forecasting System")
    
    st.markdown("""
    ## 🌍 Overview
    
    This Air Quality Forecasting System uses advanced machine learning models to predict 
    pollutant levels and provide actionable insights for better air quality management.
    
    ## 🎯 Features
    
    ### 📊 Real-time Monitoring
    - Track current air quality levels
    - Monitor multiple pollutants (PM2.5, PM10, NO2, SO2, CO, O3)
    - Calculate Air Quality Index (AQI)
    - Get health recommendations
    
    ### 🔮 Advanced Forecasting
    - Predict future pollutant levels (6-72 hours ahead)
    - XGBoost-based machine learning models
    - High accuracy predictions (R² > 0.98)
    - Multiple forecast horizons
    
    ### 📤 Data Upload & Analysis
    - Upload your own air quality data (CSV format)
    - Automatic data validation and preprocessing
    - Get instant predictions and insights
    - Download results for further analysis
    
    ### 📈 Comprehensive Analytics
    - Statistical analysis of pollutant trends
    - Correlation analysis between pollutants
    - Temporal pattern detection (hourly, daily, monthly)
    - Distribution and outlier analysis
    
    ## 🤖 Machine Learning Models
    
    ### XGBoost Regressor
    Our primary forecasting model uses XGBoost, a gradient boosting algorithm known for:
    - High accuracy on time series data
    - Efficient training and prediction
    - Robust handling of missing values
    - Feature importance analysis
    
    **Model Performance (Test Set):**
    """)
    
    # Display model metrics if available
    metrics_dir = Path('models/saved_models')
    if metrics_dir.exists():
        st.subheader("📊 Model Performance Metrics")
        
        metrics_data = []
        for pollutant in config.data_config['pollutants']:
            metrics_file = metrics_dir / f'xgboost_{pollutant.lower()}_metrics.json'
            if metrics_file.exists():
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                metrics_data.append({
                    'Pollutant': pollutant,
                    'MAE': f"{metrics['mae']:.2f}",
                    'RMSE': f"{metrics['rmse']:.2f}",
                    'MAPE': f"{metrics['mape']:.2f}%",
                    'R²': f"{metrics['r2']:.3f}"
                })
        
        if metrics_data:
            st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.markdown("""
    ## 💾 Data Sources
    
    The system supports data from:
    - Government air quality monitoring stations
    - OpenAQ API
    - Custom CSV uploads
    - Real-time sensor networks
    
    ## 📋 CSV Format Requirements
    
    For uploading your own data, ensure your CSV file includes:
    
    ```
    datetime, PM2.5, PM10, NO2, SO2, CO, O3
    2024-01-01 00:00:00, 45.2, 78.5, 32.1, 12.5, 1.2, 45.2
    2024-01-01 01:00:00, 52.1, 85.2, 35.6, 14.2, 1.4, 48.1
    ...
    ```
    
    - **datetime**: Timestamp of measurement (required)
    - **Pollutants**: At least one pollutant column (PM2.5, PM10, NO2, SO2, CO, O3)
    - **Minimum rows**: 50 for reliable predictions
    
    ## 🔧 Technical Stack
    
    - **Frontend**: Streamlit
    - **ML Framework**: XGBoost, scikit-learn
    - **Data Processing**: pandas, numpy
    - **Visualization**: Plotly, Matplotlib
    - **Configuration**: YAML
    
    ## 📖 Usage Guide
    
    ### 1. Dashboard
    View current air quality status, AQI, and recent trends.
    
    ### 2. Upload & Predict
    Upload your CSV file to get predictions and model performance metrics.
    
    ### 3. Forecasts
    Generate future predictions for any pollutant up to 72 hours ahead.
    
    ### 4. Data Analysis
    Explore patterns, correlations, and statistics in your data.
    
    ## 🚀 Getting Started
    
    1. **Train Models** (if not done):
       ```bash
       python scripts/train_models.py --pollutant all
       ```
    
    2. **Run Dashboard**:
       ```bash
       streamlit run dashboard/app.py
       ```
    
    3. **Upload Data** or use sample data to explore features
    
    ## 📞 Support
    
    For issues, questions, or contributions:
    - Check the README.md file
    - Review configuration files in `configs/`
    - Run diagnostic: `python check_models.py`
    
    ## 📜 License
    
    This project is developed for air quality monitoring and research purposes.
    
    ---
    
    **Version**: 1.0.0  
    **Last Updated**: October 2025
    """)
    
    # System status
    st.subheader("🔍 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    # Check data
    data_status = "✅ Available" if Path('data/raw/delhi_air_quality.csv').exists() else "❌ Missing"
    col1.metric("Sample Data", data_status)
    
    # Check models
    models_count = len(list(Path('models/saved_models').glob('*.pkl'))) if Path('models/saved_models').exists() else 0
    models_status = f"✅ {models_count} models" if models_count > 0 else "❌ No models"
    col2.metric("Trained Models", models_status)
    
    # Check config
    config_status = "✅ Loaded" if Path('configs/model_config.yaml').exists() else "❌ Missing"
    col3.metric("Configuration", config_status)


# Main app
def main():
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
    df = load_sample_data()
    
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
        st.metric("Location", latest.get('city', 'Delhi'))
        st.metric("Last Updated", latest['datetime'].strftime('%Y-%m-%d %H:%M'))
    
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
                st.markdown('<div class="error-box">', unsafe_allow_html=True)
                st.error("❌ Validation Failed")
                for error in errors:
                    st.write(f"- {error}")
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            # Validation passed
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("✅ CSV file validated successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
            
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
            
            # Pollutant statistics
            st.subheader("🔬 Pollutant Statistics")
            stats_data = []
            for pollutant, stats in summary['pollutants'].items():
                stats_data.append({
                    'Pollutant': pollutant,
                    'Mean': f"{stats['mean']:.2f}",
                    'Std': f"{stats['std']:.2f}",
                    'Min': f"{stats['min']:.2f}",
                    'Max': f"{stats['max']:.2f}",
                    'Missing %': f"{summary['missing_data'][pollutant]:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            # Check model compatibility
            compatibility = uploader.check_model_compatibility(df_processed)
            available_pollutants = [p for p, avail in compatibility.items() if avail]
            
            if not available_pollutants:
                st.warning("No pollutants available for prediction")
                return
            
            # Prediction section
            st.subheader("🔮 Generate Predictions")
            
            col1, col2 = st.columns(2)
            with col1:
                predict_pollutants = st.multiselect(
                    "Select pollutants to predict",
                    available_pollutants,
                    default=available_pollutants[:2] if len(available_pollutants) >= 2 else available_pollutants
                )
            
            with col2:
                include_forecast = st.checkbox("Include future forecast", value=False)
                if include_forecast:
                    forecast_hours = st.slider("Forecast hours ahead", 6, 72, 24)
            
            if st.button("🚀 Run Predictions", type="primary"):
                if not predict_pollutants:
                    st.warning("Please select at least one pollutant")
                    return
                
                # Initialize prediction handler
                handler = PredictionHandler()
                
                # Load models
                with st.spinner("Loading models..."):
                    loaded_models = handler.load_models(predict_pollutants)
                    
                    failed_models = [p for p, success in loaded_models.items() if not success]
                    if failed_models:
                        st.warning(f"Could not load models for: {', '.join(failed_models)}")
                    
                    success_models = [p for p, success in loaded_models.items() if success]
                    if not success_models:
                        st.error("No models could be loaded. Please train models first.")
                        return
                
                # Make predictions
                with st.spinner("Generating predictions..."):
                    results = handler.predict_all_pollutants(df_processed)
                    st.session_state.prediction_results = results
                
                # Calculate metrics
                metrics = handler.calculate_metrics(results)
                
                # Display results
                st.success("✅ Predictions completed!")
                
                # Show metrics
                st.subheader("📈 Model Performance")
                metrics_data = []
                for pollutant, metric in metrics.items():
                    metrics_data.append({
                        'Pollutant': pollutant,
                        'MAE': f"{metric['mae']:.2f}",
                        'RMSE': f"{metric['rmse']:.2f}",
                        'MAPE': f"{metric['mape']:.2f}%",
                        'R²': f"{metric['r2']:.3f}",
                        'Samples': metric['samples']
                    })
                
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
                
                # Visualize predictions
                st.subheader("📊 Prediction Visualizations")
                
                for pollutant in success_models:
                    if pollutant in results and results[pollutant] is not None:
                        with st.expander(f"📈 {pollutant} Results", expanded=True):
                            result_df = results[pollutant]
                            
                            # Plot comparison
                            st.plotly_chart(
                                plot_prediction_comparison(result_df, pollutant),
                                use_container_width=True
                            )
                            
                            # Error distribution
                            fig_error = px.histogram(
                                result_df,
                                x=f'{pollutant}_error',
                                title=f'{pollutant} - Prediction Error Distribution',
                                labels={f'{pollutant}_error': 'Prediction Error (μg/m³)'}
                            )
                            st.plotly_chart(fig_error, use_container_width=True)
                            
                            # Show sample predictions
                            st.write("**Sample Predictions:**")
                            display_cols = [f'{pollutant}_actual', f'{pollutant}_predicted', f'{pollutant}_error']
                            st.dataframe(result_df[display_cols].head(10), use_container_width=True)
                
                # Generate insights
                st.subheader("💡 Insights")
                insights = handler.generate_insights(df_processed, results)
                
                for insight in insights:
                    st.write(insight)
                
                # Future forecast
                if include_forecast:
                    st.subheader("🔮 Future Forecast")
                    
                    forecast_pollutant = st.selectbox(
                        "Select pollutant for forecast",
                        success_models
                    )
                    
                    if st.button("Generate Forecast"):
                        with st.spinner("Generating forecast..."):
                            forecast_df = handler.forecast_future(
                                df_processed,
                                forecast_pollutant,
                                hours_ahead=forecast_hours
                            )
                        
                        # Plot forecast
                        fig_forecast = go.Figure()
                        
                        # Historical data (last 48 hours)
                        historical = df_processed[forecast_pollutant].tail(48)
                        fig_forecast.add_trace(go.Scatter(
                            x=historical.index,
                            y=historical.values,
                            mode='lines',
                            name='Historical',
                            line=dict(color='blue')
                        ))
                        
                        # Forecast
                        fig_forecast.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df[f'{forecast_pollutant}_forecast'],
                            mode='lines+markers',
                            name='Forecast',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_forecast.update_layout(
                            title=f'{forecast_pollutant} - {forecast_hours}h Forecast',
                            xaxis_title='Date',
                            yaxis_title=f'{forecast_pollutant} (μg/m³)',
                            hovermode='x unified',
                            height=400
                        )
                        
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast statistics
                        avg_forecast = forecast_df[f'{forecast_pollutant}_forecast'].mean()
                        max_forecast = forecast_df[f'{forecast_pollutant}_forecast'].max()
                        min_forecast = forecast_df[f'{forecast_pollutant}_forecast'].min()
                        
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Average", f"{avg_forecast:.1f} μg/m³")
                        col2.metric("Maximum", f"{max_forecast:.1f} μg/m³")
                        col3.metric("Minimum", f"{min_forecast:.1f} μg/m³")
                
                # Download results
                st.subheader("💾 Download Results")
                
                # Combine all results
                combined_results = df_processed.copy()
                for pollutant, result_df in results.items():
                    if result_df is not None:
                        combined_results = combined_results.join(
                            result_df[[f'{pollutant}_predicted']],
                            how='left'
                        )
                
                csv = combined_results.to_csv()
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)
    
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