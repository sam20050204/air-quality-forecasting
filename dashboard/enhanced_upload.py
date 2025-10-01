"""
Enhanced Upload & Prediction Page with Comprehensive Data Analysis
Save as: dashboard/enhanced_upload.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.api_calculator import AQICalculator
from src.models.xgboost_model import XGBoostModel
from src.data.data_uploader import CSVUploader
from src.data.prediction_handler import PredictionHandler

def show_enhanced_upload_page():
    """Enhanced upload page with comprehensive analysis"""
    st.header("📤 Upload & Analyze Air Quality Data")
    
    # Initialize session state
    if 'uploaded_df' not in st.session_state:
        st.session_state.uploaded_df = None
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    
    # File uploader
    st.subheader("📁 Step 1: Upload Your CSV File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file with air quality data",
            type=['csv'],
            help="Upload a CSV file containing datetime and pollutant columns"
        )
    
    with col2:
        st.info("""
        **Required Format:**
        - datetime column
        - At least one pollutant: PM2.5, PM10, NO2, SO2, CO, O3
        - Minimum 50 rows
        """)
    
    # Show example format
    with st.expander("📋 View Example CSV Format"):
        example_df = pd.DataFrame({
            'datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
            'PM2.5': [45.2, 52.1, 48.5],
            'PM10': [78.5, 85.2, 80.1],
            'NO2': [32.1, 35.6, 33.2],
            'SO2': [12.5, 14.2, 13.1],
            'CO': [1.2, 1.5, 1.3],
            'O3': [45.2, 48.5, 46.8]
        })
        st.dataframe(example_df, use_container_width=True)
        
        # Download example CSV
        csv = example_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Example CSV",
            data=csv,
            file_name="example_air_quality.csv",
            mime="text/csv"
        )
    
    # Process uploaded file
    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)
            
            # Initialize uploader
            uploader = CSVUploader()
            
            # Validate CSV
            is_valid, errors = uploader.validate_csv(df)
            
            if not is_valid:
                st.error("❌ CSV Validation Failed")
                for error in errors:
                    st.warning(f"⚠️ {error}")
                return
            
            # Success message
            st.success("✅ File uploaded and validated successfully!")
            
            # Preprocess data
            with st.spinner("Processing data..."):
                processed_df = uploader.preprocess_uploaded_data(df)
                st.session_state.uploaded_df = processed_df
                st.session_state.analysis_complete = True
            
            # Display analysis
            show_data_analysis(processed_df, uploader)
            
            # Prediction section
            st.markdown("---")
            show_prediction_section(processed_df)
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please check that your CSV file is properly formatted")
    
    else:
        st.info("👆 Upload a CSV file to begin analysis")


def show_data_analysis(df: pd.DataFrame, uploader: CSVUploader):
    """Show comprehensive data analysis"""
    st.subheader("📊 Step 2: Data Analysis Report")
    
    # Get summary
    summary = uploader.get_data_summary(df)
    
    # Overview metrics
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{summary['total_rows']:,}")
    
    with col2:
        date_range_days = (pd.to_datetime(summary['date_range']['end']) - 
                          pd.to_datetime(summary['date_range']['start'])).days
        st.metric("Time Span", f"{date_range_days} days")
    
    with col3:
        st.metric("Pollutants", len(summary['pollutants']))
    
    with col4:
        quality_color = {"Good": "green", "Fair": "orange", "Poor": "red"}
        st.metric("Data Quality", summary['data_quality'])
    
    # Date range
    st.markdown(f"**Date Range:** {summary['date_range']['start']} to {summary['date_range']['end']}")
    
    # Pollutant statistics
    st.markdown("### 🔬 Pollutant Statistics")
    
    stats_data = []
    for pollutant, stats in summary['pollutants'].items():
        stats_data.append({
            'Pollutant': pollutant,
            'Mean': f"{stats['mean']:.2f}",
            'Std Dev': f"{stats['std']:.2f}",
            'Min': f"{stats['min']:.2f}",
            'Max': f"{stats['max']:.2f}",
            'Median': f"{stats['median']:.2f}",
            'Missing %': f"{summary['missing_data'][pollutant]:.1f}%"
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown("### 📉 Data Visualizations")
    
    tabs = st.tabs([
        "📊 Time Series", 
        "📈 Distributions", 
        "🔗 Correlations", 
        "📅 Patterns",
        "⚠️ Data Quality"
    ])
    
    pollutants = list(summary['pollutants'].keys())
    
    # Tab 1: Time Series
    with tabs[0]:
        st.markdown("**Pollutant Trends Over Time**")
        selected_pollutant = st.selectbox("Select Pollutant", pollutants, key="ts_pollutant")
        
        fig = px.line(df.reset_index(), x='datetime', y=selected_pollutant,
                     title=f'{selected_pollutant} Over Time')
        fig.update_layout(
            xaxis_title='Date',
            yaxis_title=f'{selected_pollutant} (μg/m³)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics for selected pollutant
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average", f"{df[selected_pollutant].mean():.2f} μg/m³")
        with col2:
            st.metric("Peak Value", f"{df[selected_pollutant].max():.2f} μg/m³")
        with col3:
            st.metric("Variability", f"{df[selected_pollutant].std():.2f} μg/m³")
    
    # Tab 2: Distributions
    with tabs[1]:
        st.markdown("**Pollutant Distribution Analysis**")
        
        dist_pollutant = st.selectbox("Select Pollutant", pollutants, key="dist_pollutant")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=dist_pollutant, nbins=50,
                             title=f'{dist_pollutant} Distribution')
            fig.update_layout(
                xaxis_title=f'{dist_pollutant} (μg/m³)',
                yaxis_title='Frequency',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=dist_pollutant,
                        title=f'{dist_pollutant} Box Plot')
            fig.update_layout(
                yaxis_title=f'{dist_pollutant} (μg/m³)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Percentile analysis
        st.markdown("**Percentile Analysis**")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        perc_values = [df[dist_pollutant].quantile(p/100) for p in percentiles]
        
        perc_df = pd.DataFrame({
            'Percentile': [f'{p}th' for p in percentiles],
            'Value (μg/m³)': [f'{v:.2f}' for v in perc_values]
        })
        st.dataframe(perc_df, use_container_width=True, hide_index=True)
    
    # Tab 3: Correlations
    with tabs[2]:
        st.markdown("**Pollutant Correlation Matrix**")
        
        # Calculate correlation
        corr_matrix = df[pollutants].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation"),
                       x=pollutants,
                       y=pollutants,
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       zmin=-1, zmax=1)
        fig.update_layout(
            title='Correlation Heatmap',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation insights
        st.markdown("**Key Correlations:**")
        
        # Find strong correlations
        for i, poll1 in enumerate(pollutants):
            for j, poll2 in enumerate(pollutants):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[poll1, poll2]
                    if abs(corr_val) > 0.7:
                        emoji = "🔴" if corr_val > 0 else "🔵"
                        st.write(f"{emoji} **{poll1}** and **{poll2}**: {corr_val:.2f} "
                               f"({'Strong positive' if corr_val > 0 else 'Strong negative'} correlation)")
    
    # Tab 4: Patterns
    with tabs[3]:
        st.markdown("**Temporal Patterns Analysis**")
        
        pattern_pollutant = st.selectbox("Select Pollutant", pollutants, key="pattern_pollutant")
        
        # Add time-based features
        df_patterns = df.reset_index()
        df_patterns['hour'] = df_patterns['datetime'].dt.hour
        df_patterns['day_of_week'] = df_patterns['datetime'].dt.day_name()
        df_patterns['month'] = df_patterns['datetime'].dt.month_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Hourly pattern
            hourly_avg = df_patterns.groupby('hour')[pattern_pollutant].mean()
            
            fig = px.line(x=hourly_avg.index, y=hourly_avg.values,
                         title=f'{pattern_pollutant} - Hourly Pattern')
            fig.update_layout(
                xaxis_title='Hour of Day',
                yaxis_title=f'{pattern_pollutant} (μg/m³)',
                showlegend=False
            )
            fig.update_traces(mode='lines+markers')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Day of week pattern
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            dow_avg = df_patterns.groupby('day_of_week')[pattern_pollutant].mean().reindex(dow_order)
            
            fig = px.bar(x=dow_avg.index, y=dow_avg.values,
                        title=f'{pattern_pollutant} - Day of Week Pattern')
            fig.update_layout(
                xaxis_title='Day of Week',
                yaxis_title=f'{pattern_pollutant} (μg/m³)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Identify peak hours
        peak_hour = hourly_avg.idxmax()
        low_hour = hourly_avg.idxmin()
        
        st.info(f"📊 **Pattern Insights:**\n\n"
               f"- Peak pollution typically occurs at **{peak_hour}:00** ({hourly_avg[peak_hour]:.2f} μg/m³)\n"
               f"- Lowest levels at **{low_hour}:00** ({hourly_avg[low_hour]:.2f} μg/m³)\n"
               f"- Peak-to-low variation: **{((hourly_avg[peak_hour] - hourly_avg[low_hour]) / hourly_avg[low_hour] * 100):.1f}%**")
    
    # Tab 5: Data Quality
    with tabs[4]:
        st.markdown("**Data Quality Assessment**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Missing data visualization
            missing_data = []
            for pollutant in pollutants:
                missing_pct = (df[pollutant].isnull().sum() / len(df)) * 100
                missing_data.append({
                    'Pollutant': pollutant,
                    'Missing %': missing_pct
                })
            
            missing_df = pd.DataFrame(missing_data)
            
            fig = px.bar(missing_df, x='Pollutant', y='Missing %',
                        title='Missing Data by Pollutant')
            fig.update_layout(
                yaxis_title='Missing Data (%)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Outlier detection
            outlier_counts = []
            for pollutant in pollutants:
                Q1 = df[pollutant].quantile(0.25)
                Q3 = df[pollutant].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((df[pollutant] < lower) | (df[pollutant] > upper)).sum()
                outlier_counts.append({
                    'Pollutant': pollutant,
                    'Outliers': outliers,
                    'Percentage': (outliers / len(df)) * 100
                })
            
            outlier_df = pd.DataFrame(outlier_counts)
            
            fig = px.bar(outlier_df, x='Pollutant', y='Percentage',
                        title='Potential Outliers by Pollutant')
            fig.update_layout(
                yaxis_title='Outliers (%)',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality summary
        total_cells = len(df) * len(pollutants)
        total_missing = sum([summary['missing_data'][p] * len(df) / 100 for p in pollutants])
        completeness = ((total_cells - total_missing) / total_cells) * 100
        
        st.success(f"✅ **Data Completeness: {completeness:.1f}%**")
        
        if completeness >= 95:
            st.info("🎯 Excellent data quality - suitable for accurate predictions")
        elif completeness >= 85:
            st.warning("⚠️ Good data quality - minor gaps have been interpolated")
        else:
            st.error("❗ Data quality concerns - predictions may be less reliable")
    
    # Download processed data
    st.markdown("---")
    st.markdown("### 💾 Download Analysis Report")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download processed CSV
        csv = df.to_csv()
        st.download_button(
            label="📥 Download Processed Data (CSV)",
            data=csv,
            file_name=f"processed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download statistics JSON
        json_data = json.dumps(summary, indent=2, default=str)
        st.download_button(
            label="📥 Download Statistics (JSON)",
            data=json_data,
            file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Download full report (text)
        report = generate_text_report(df, summary)
        st.download_button(
            label="📥 Download Full Report (TXT)",
            data=report,
            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )


def show_prediction_section(df: pd.DataFrame):
    """Show prediction section with model-based forecasts"""
    st.subheader("🔮 Step 3: Generate Predictions")
    
    # Check if models are available
    model_dir = Path('models/saved_models')
    models_exist = model_dir.exists() and len(list(model_dir.glob('*.pkl'))) > 0
    
    if not models_exist:
        st.warning("⚠️ Trained models not found. Predictions are not available.")
        st.info("To enable predictions, train models first:\n```bash\npython scripts/train_models.py --pollutant all\n```")
        return
    
    # Initialize prediction handler
    handler = PredictionHandler()
    
    # Load models
    pollutants_in_data = [p for p in config.data_config['pollutants'] if p in df.columns]
    
    with st.spinner("Loading prediction models..."):
        loaded_models = handler.load_models(pollutants_in_data)
    
    available_models = [p for p, loaded in loaded_models.items() if loaded]
    
    if not available_models:
        st.error("❌ No models could be loaded for the available pollutants")
        return
    
    st.success(f"✅ Loaded models for: {', '.join(available_models)}")
    
    # Prediction options
    col1, col2 = st.columns(2)
    
    with col1:
        prediction_type = st.radio(
            "Prediction Type",
            ["Validate Historical Data", "Forecast Future Values"],
            help="Choose whether to validate against existing data or forecast future values"
        )
    
    with col2:
        if prediction_type == "Forecast Future Values":
            forecast_hours = st.slider("Forecast Horizon (hours)", 1, 168, 24)
    
    # Run predictions
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Generating predictions..."):
            if prediction_type == "Validate Historical Data":
                # Make predictions on historical data
                results = handler.predict_all_pollutants(df)
                
                # Calculate metrics
                metrics = handler.calculate_metrics(results)
                
                # Display results
                show_prediction_results(results, metrics, df)
                
                # Store in session state
                st.session_state.prediction_results = {
                    'type': 'historical',
                    'results': results,
                    'metrics': metrics
                }
                
            else:  # Forecast future
                # Generate forecasts
                forecast_results = {}
                for pollutant in available_models:
                    try:
                        forecast_df = handler.forecast_future(df, pollutant, forecast_hours)
                        forecast_results[pollutant] = forecast_df
                    except Exception as e:
                        st.warning(f"Could not forecast {pollutant}: {str(e)}")
                
                # Display forecasts
                show_forecast_results(forecast_results, df)
                
                # Store in session state
                st.session_state.prediction_results = {
                    'type': 'forecast',
                    'results': forecast_results
                }


def show_prediction_results(results: dict, metrics: dict, original_df: pd.DataFrame):
    """Display prediction results and metrics"""
    st.markdown("### 📊 Prediction Results")
    
    # Metrics summary
    st.markdown("#### 🎯 Model Performance Metrics")
    
    metric_cols = st.columns(len(metrics))
    
    for idx, (pollutant, metric_dict) in enumerate(metrics.items()):
        with metric_cols[idx]:
            st.metric(
                f"{pollutant}",
                f"R² = {metric_dict['r2']:.3f}",
                help=f"RMSE: {metric_dict['rmse']:.2f}, MAE: {metric_dict['mae']:.2f}"
            )
    
    # Detailed metrics table
    with st.expander("📈 View Detailed Metrics"):
        metrics_data = []
        for pollutant, m in metrics.items():
            metrics_data.append({
                'Pollutant': pollutant,
                'R² Score': f"{m['r2']:.4f}",
                'RMSE': f"{m['rmse']:.2f}",
                'MAE': f"{m['mae']:.2f}",
                'MAPE': f"{m['mape']:.2f}%",
                'Samples': m['samples']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    # Visualizations
    st.markdown("#### 📉 Actual vs Predicted")
    
    selected_pollutant = st.selectbox("Select Pollutant to Visualize", list(results.keys()))
    
    if results[selected_pollutant] is not None:
        result_df = results[selected_pollutant]
        
        # Time series comparison
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df[f'{selected_pollutant}_actual'],
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=result_df.index,
            y=result_df[f'{selected_pollutant}_predicted'],
            mode='lines',
            name='Predicted',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{selected_pollutant} - Actual vs Predicted',
            xaxis_title='Date',
            yaxis_title=f'{selected_pollutant} (μg/m³)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                result_df,
                x=f'{selected_pollutant}_actual',
                y=f'{selected_pollutant}_predicted',
                title='Prediction Accuracy',
                labels={
                    f'{selected_pollutant}_actual': 'Actual',
                    f'{selected_pollutant}_predicted': 'Predicted'
                }
            )
            
            # Add diagonal line
            min_val = min(result_df[f'{selected_pollutant}_actual'].min(),
                         result_df[f'{selected_pollutant}_predicted'].min())
            max_val = max(result_df[f'{selected_pollutant}_actual'].max(),
                         result_df[f'{selected_pollutant}_predicted'].max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', dash='dash')
            ))
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error distribution
            fig = px.histogram(
                result_df,
                x=f'{selected_pollutant}_error',
                title='Prediction Error Distribution',
                labels={f'{selected_pollutant}_error': 'Error (μg/m³)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download predictions
        st.markdown("#### 💾 Download Predictions")
        
        # Combine all results
        combined_df = original_df.copy()
        for pollutant, result in results.items():
            if result is not None:
                combined_df[f'{pollutant}_predicted'] = result[f'{pollutant}_predicted']
        
        csv = combined_df.to_csv()
        st.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def show_forecast_results(forecasts: dict, historical_df: pd.DataFrame):
    """Display forecast results"""
    st.markdown("### 🔮 Future Forecasts")
    
    selected_pollutant = st.selectbox("Select Pollutant", list(forecasts.keys()))
    
    if selected_pollutant in forecasts:
        forecast_df = forecasts[selected_pollutant]
        
        # Combine historical and forecast
        historical_tail = historical_df[[selected_pollutant]].tail(168)  # Last week
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_tail.index,
            y=historical_tail[selected_pollutant],
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[f'{selected_pollutant}_forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{selected_pollutant} - Forecast',
            xaxis_title='Date',
            yaxis_title=f'{selected_pollutant} (μg/m³)',
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Forecast Mean", f"{forecast_df[f'{selected_pollutant}_forecast'].mean():.2f} μg/m³")
        
        with col2:
            st.metric("Forecast Peak", f"{forecast_df[f'{selected_pollutant}_forecast'].max():.2f} μg/m³")
        
        with col3:
            trend = forecast_df[f'{selected_pollutant}_forecast'].iloc[-1] - forecast_df[f'{selected_pollutant}_forecast'].iloc[0]
            st.metric("Overall Trend", f"{'+' if trend > 0 else ''}{trend:.2f} μg/m³")
        
        # Download forecast
        csv = forecast_df.to_csv()
        st.download_button(
            label="📥 Download Forecast (CSV)",
            data=csv,
            file_name=f"forecast_{selected_pollutant}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def generate_text_report(df: pd.DataFrame, summary: dict) -> str:
    """Generate comprehensive text report"""
    report = []
    report.append("=" * 80)
    report.append("AIR QUALITY DATA ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Dataset overview
    report.append("\n" + "=" * 80)
    report.append("1. DATASET OVERVIEW")
    report.append("=" * 80)
    report.append(f"Total Records: {summary['total_rows']:,}")
    report.append(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    report.append(f"Data Quality: {summary['data_quality']}")
    report.append(f"Pollutants Monitored: {len(summary['pollutants'])}")
    
    # Pollutant statistics
    report.append("\n" + "=" * 80)
    report.append("2. POLLUTANT STATISTICS")
    report.append("=" * 80)
    
    for pollutant, stats in summary['pollutants'].items():
        report.append(f"\n{pollutant}:")
        report.append(f"  Mean: {stats['mean']:.2f} μg/m³")
        report.append(f"  Std Dev: {stats['std']:.2f} μg/m³")
        report.append(f"  Min: {stats['min']:.2f} μg/m³")
        report.append(f"  Max: {stats['max']:.2f} μg/m³")
        report.append(f"  Median: {stats['median']:.2f} μg/m³")
        report.append(f"  Missing Data: {summary['missing_data'][pollutant]:.1f}%")
    
    # Data quality assessment
    report.append("\n" + "=" * 80)
    report.append("3. DATA QUALITY ASSESSMENT")
    report.append("=" * 80)
    
    for pollutant in summary['pollutants'].keys():
        missing_pct = summary['missing_data'][pollutant]
        status = "✓ Good" if missing_pct < 5 else "⚠ Fair" if missing_pct < 15 else "✗ Poor"
        report.append(f"{pollutant}: {status} ({missing_pct:.1f}% missing)")
    
    # Key insights
    report.append("\n" + "=" * 80)
    report.append("4. KEY INSIGHTS")
    report.append("=" * 80)
    
    # Find most polluted
    max_pollutant = max(summary['pollutants'].items(), key=lambda x: x[1]['mean'])
    report.append(f"• Highest average concentration: {max_pollutant[0]} ({max_pollutant[1]['mean']:.2f} μg/m³)")
    
    # Find most variable
    max_std_pollutant = max(summary['pollutants'].items(), key=lambda x: x[1]['std'])
    report.append(f"• Most variable pollutant: {max_std_pollutant[0]} (σ = {max_std_pollutant[1]['std']:.2f})")
    
    # Data completeness
    total_missing = sum(summary['missing_data'].values()) / len(summary['missing_data'])
    report.append(f"• Average data completeness: {100 - total_missing:.1f}%")
    
    # Recommendations
    report.append("\n" + "=" * 80)
    report.append("5. RECOMMENDATIONS")
    report.append("=" * 80)
    
    if summary['data_quality'] == 'Good':
        report.append("• Data quality is excellent for predictive modeling")
        report.append("• Proceed with confidence for forecasting applications")
    elif summary['data_quality'] == 'Fair':
        report.append("• Data quality is acceptable with minor gaps")
        report.append("• Consider additional data collection for improved accuracy")
    else:
        report.append("• Data quality concerns detected")
        report.append("• Recommend data validation and additional collection")
    
    if total_missing > 10:
        report.append("• Significant missing data detected - use interpolated values with caution")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)