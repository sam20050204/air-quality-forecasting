"""
Standalone Air Quality Dashboard - No dependencies on project structure
Save as: simple_app.py
Run with: streamlit run simple_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Air Quality Forecasting",
    page_icon="🌍",
    layout="wide"
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
    </style>
""", unsafe_allow_html=True)

# Generate or load sample data
@st.cache_data
def load_data():
    """Load or generate sample air quality data"""
    data_path = Path('data/raw/delhi_air_quality.csv')
    
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            return df
        except:
            pass
    
    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=168*4, freq='H')  # 4 weeks
    
    hours = dates.hour.values
    days = np.arange(len(dates))
    
    df = pd.DataFrame({
        'datetime': dates,
        'city': 'Delhi',
        'PM2.5': 60 + 40*np.sin(days/24*2*np.pi/7) + 15*np.sin(hours/24*2*np.pi) + np.random.normal(0, 10, len(dates)),
        'PM10': 95 + 60*np.sin(days/24*2*np.pi/7) + 25*np.sin(hours/24*2*np.pi) + np.random.normal(0, 15, len(dates)),
        'NO2': 35 + 15*np.sin(days/24*2*np.pi/7) + 8*np.sin(hours/24*2*np.pi) + np.random.normal(0, 5, len(dates)),
        'SO2': 18 + 8*np.sin(days/24*2*np.pi/7) + 4*np.sin(hours/24*2*np.pi) + np.random.normal(0, 3, len(dates)),
        'CO': 1.8 + 0.9*np.sin(days/24*2*np.pi/7) + 0.4*np.sin(hours/24*2*np.pi) + np.random.normal(0, 0.2, len(dates)),
        'O3': 52 + 25*np.sin(days/24*2*np.pi/7) + 18*np.sin(hours/24*2*np.pi) + np.random.normal(0, 6, len(dates)),
    })
    
    # Ensure positive values
    for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
        df[col] = df[col].clip(lower=0.1)
    
    return df

def calculate_aqi(pm25):
    """Calculate AQI from PM2.5"""
    breakpoints = [
        (0, 12, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 500, 301, 500)
    ]
    
    for bp_lo, bp_hi, aqi_lo, aqi_hi in breakpoints:
        if bp_lo <= pm25 <= bp_hi:
            aqi = ((aqi_hi - aqi_lo) / (bp_hi - bp_lo)) * (pm25 - bp_lo) + aqi_lo
            return round(aqi)
    return 500

def get_aqi_category(aqi):
    """Get AQI category and color"""
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"

def create_gauge_chart(value, title):
    """Create gauge chart for AQI"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 500]},
            'bar': {'color': "darkblue"},
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

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🌍 Air Quality Forecasting System</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select Page", [
        "📊 Dashboard",
        "📈 Analysis",
        "📤 Upload Data",
        "ℹ️ About"
    ])
    
    # Load data
    df = load_data()
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "📈 Analysis":
        show_analysis(df)
    elif page == "📤 Upload Data":
        show_upload(df)
    else:
        show_about()

def show_dashboard(df):
    """Main dashboard"""
    st.header("📊 Current Air Quality Dashboard")
    
    # Get latest data
    latest = df.iloc[-1]
    current_aqi = calculate_aqi(latest['PM2.5'])
    category, color = get_aqi_category(current_aqi)
    
    # Display current status
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
        st.success("✅ Air quality is good. Enjoy outdoor activities!")
    elif current_aqi <= 100:
        st.info("ℹ️ Air quality is moderate. Sensitive individuals should consider reducing prolonged outdoor exertion.")
    elif current_aqi <= 150:
        st.warning("⚠️ Unhealthy for sensitive groups. Reduce prolonged or heavy outdoor exertion.")
    else:
        st.error("🚨 Air quality is unhealthy. Avoid outdoor activities and stay indoors.")
    
    # Recent trends
    st.subheader("📈 Recent Trends (Last 7 Days)")
    
    recent_df = df.tail(168)
    pollutant = st.selectbox("Select Pollutant", ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3'])
    
    fig = px.line(recent_df, x='datetime', y=pollutant, title=f'{pollutant} Levels Over Time')
    fig.update_layout(xaxis_title='Date', yaxis_title=f'{pollutant} (μg/m³)', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

def show_analysis(df):
    """Analysis page"""
    st.header("📈 Data Analysis")
    
    # Summary statistics
    st.subheader("📊 Summary Statistics")
    
    pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
    stats_df = df[pollutants].describe().T
    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔗 Pollutant Correlations")
    
    corr = df[pollutants].corr()
    fig = px.imshow(corr, 
                    labels=dict(color="Correlation"),
                    x=pollutants, y=pollutants,
                    color_continuous_scale='RdBu_r',
                    aspect="auto")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution
    st.subheader("📊 Pollutant Distribution")
    
    pollutant = st.selectbox("Select pollutant", pollutants)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x=pollutant, nbins=50, title=f"{pollutant} Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=pollutant, title=f"{pollutant} Box Plot")
        st.plotly_chart(fig, use_container_width=True)
    
    # Hourly pattern
    st.subheader("⏰ Hourly Pattern")
    
    df_hourly = df.copy()
    df_hourly['hour'] = pd.to_datetime(df_hourly['datetime']).dt.hour
    hourly_avg = df_hourly.groupby('hour')[pollutant].mean()
    
    fig = px.line(x=hourly_avg.index, y=hourly_avg.values, 
                  title=f"{pollutant} - Average by Hour of Day",
                  labels={'x': 'Hour', 'y': f'{pollutant} (μg/m³)'})
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

def show_upload(df):
    """Upload page"""
    st.header("📤 Upload Your Data")
    
    st.markdown("""
    Upload your own air quality CSV file for analysis.
    
    **Required format:**
    - datetime column (any format)
    - At least one pollutant: PM2.5, PM10, NO2, SO2, CO, O3
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            
            st.success("✅ File uploaded successfully!")
            st.subheader("Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            st.subheader("Data Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Rows", len(df_upload))
            col2.metric("Columns", len(df_upload.columns))
            col3.metric("Size", f"{uploaded_file.size / 1024:.1f} KB")
            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Show example
        st.subheader("Example Format")
        example = pd.DataFrame({
            'datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00'],
            'PM2.5': [45.2, 52.1],
            'PM10': [78.5, 85.2],
            'NO2': [32.1, 35.6]
        })
        st.dataframe(example, use_container_width=True)

def show_about():
    """About page"""
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🌍 Air Quality Forecasting System
    
    This system provides real-time air quality monitoring and analysis.
    
    ### Features:
    - 📊 **Dashboard**: Current AQI and pollutant levels
    - 📈 **Analysis**: Statistical analysis and trends
    - 📤 **Upload**: Analyze your own data
    - 🔮 **Forecasting**: ML-based predictions (full version)
    
    ### Pollutants Monitored:
    - **PM2.5**: Fine particulate matter (≤2.5 μm)
    - **PM10**: Coarse particulate matter (≤10 μm)
    - **NO2**: Nitrogen dioxide
    - **SO2**: Sulfur dioxide
    - **CO**: Carbon monoxide
    - **O3**: Ozone
    
    ### AQI Categories:
    - 🟢 **0-50**: Good
    - 🟡 **51-100**: Moderate
    - 🟠 **101-150**: Unhealthy for Sensitive Groups
    - 🔴 **151-200**: Unhealthy
    - 🟣 **201-300**: Very Unhealthy
    - 🔴 **301-500**: Hazardous
    
    ### Data Sources:
    Currently using sample data. For real data:
    - Government monitoring stations
    - OpenAQ API
    - Custom CSV uploads
    
    ### Version: 1.0.0
    """)
    
    st.success("✅ System is operational!")

if __name__ == "__main__":
    main()