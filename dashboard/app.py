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