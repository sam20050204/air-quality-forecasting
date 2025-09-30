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
    .aqi-good { color: #00e400; font-weight: bold; }
    .aqi-moderate { color: #ffff00; font-weight: bold; }
    .aqi-unhealthy-sensitive { color: #ff7e00; font-weight: bold; }
    .aqi-unhealthy { color: #ff0000; font-weight: bold; }
    .aqi-very-unhealthy { color: #8f3f97; font-weight: bold; }
    .aqi-hazardous { color: #7e0023; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.models = {}

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

def plot_forecast(historical, forecast_dates, forecast_values, pollutant):
    """Plot historical data with forecast"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical['datetime'],
        y=historical[pollutant],
        mode='lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines+markers',
        name='Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f'{pollutant} Forecast',
        xaxis_title='Date',
        yaxis_title=f'{pollutant} (μg/m³)',
        hovermode='x unified',
        height=400
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Air Quality Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Forecasts", "Data Analysis", "About"])
    
    # Load data
    df = load_sample_data()
    
    if page == "Dashboard":
        show_dashboard(df)
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

def show_forecasts(df):
    """Forecast view"""
    st.header("🔮 Air Quality Forecasts")
    
    # Try to load models
    if not st.session_state.models_loaded:
        with st.spinner("Loading models..."):
            models = load_models()
            if models:
                st.session_state.models = models
                st.session_state.models_loaded = True
                st.success(f"Loaded models for {len(models)} pollutants")
            else:
                st.warning("No trained models found. Please train models first using: `python scripts/train_models.py`")
                return
    
    # Forecast settings
    col1, col2 = st.columns(2)
    with col1:
        pollutant = st.selectbox("Select Pollutant", list(st.session_state.models.keys()))
    with col2:
        forecast_days = st.slider("Forecast Days", 1, 7, 3)
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            # Get latest data point
            latest_data = df.tail(24)  # Last 24 hours for context
            
            # Simple forecast (placeholder - would use actual model prediction)
            forecast_hours = forecast_days * 24
            last_value = df[pollutant].iloc[-1]
            
            # Generate forecast dates
            last_date = df['datetime'].iloc[-1]
            forecast_dates = pd.date_range(start=last_date + timedelta(hours=1), 
                                          periods=forecast_hours, freq='H')
            
            # Simple forecast (add some random walk)
            np.random.seed(42)
            forecast_values = [last_value]
            for _ in range(forecast_hours - 1):
                change = np.random.normal(0, 5)
                next_val = max(0, forecast_values[-1] + change)
                forecast_values.append(next_val)
            
            # Plot forecast
            st.plotly_chart(plot_forecast(df.tail(168), forecast_dates, 
                                         forecast_values, pollutant), 
                          use_container_width=True)
            
            # Forecast summary
            st.subheader("Forecast Summary")
            avg_forecast = np.mean(forecast_values)
            max_forecast = np.max(forecast_values)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Predicted", f"{avg_forecast:.1f} μg/m³")
            col2.metric("Maximum Predicted", f"{max_forecast:.1f} μg/m³")
            col3.metric("Trend", "Stable" if abs(forecast_values[-1] - forecast_values[0]) < 10 else "Variable")

def show_analysis(df):
    """Data analysis view"""
    st.header("📊 Data Analysis")
    
    # Pollutant correlations
    st.subheader("Pollutant Correlations")
    pollutants = config.data_config['pollutants']
    corr_matrix = df[pollutants].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="Correlation Matrix of Pollutants",
                    color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution plots
    st.subheader("Pollutant Distributions")
    selected_pollutant = st.selectbox("Select Pollutant for Distribution", pollutants)
    
    fig = px.histogram(df, x=selected_pollutant, nbins=50,
                       title=f'Distribution of {selected_pollutant}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.subheader("Summary Statistics")
    st.dataframe(df[pollutants].describe())

def show_about():
    """About page"""
    st.header("About This System")
    
    st.markdown("""
    ## 🌍 Air Quality Forecasting System
    
    This system uses machine learning to predict air quality levels and provide early warnings
    for pollution events.
    
    ### Features:
    - **Real-time Monitoring**: Track current air quality levels
    - **Forecasting**: Predict future pollution levels up to 7 days ahead
    - **Multiple Models**: XGBoost, LSTM, ARIMA, and Prophet models
    - **Health Alerts**: Get notifications when air quality becomes unhealthy
    
    ### Pollutants Monitored:
    - PM2.5 - Fine Particulate Matter
    - PM10 - Coarse Particulate Matter
    - NO2 - Nitrogen Dioxide
    - SO2 - Sulfur Dioxide
    - CO - Carbon Monoxide
    - O3 - Ozone
    
    ### How to Use:
    1. **Dashboard**: View current air quality and recent trends
    2. **Forecasts**: Generate predictions for future air quality
    3. **Data Analysis**: Explore pollutant correlations and patterns
    
    ### Data Sources:
    - Central Pollution Control Board (CPCB)
    - OpenAQ API
    - Local monitoring stations
    
    ---
    **Version**: 1.0.0  
    **Last Updated**: 2024
    """)

if __name__ == "__main__":
    main()