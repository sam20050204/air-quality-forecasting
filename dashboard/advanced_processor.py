"""
Advanced Data Cleaning, Preprocessing & Visualization System
Handles raw, messy CSV files and produces clean, analyzed data
Save as: dashboard/advanced_processor.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta
import json
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.api_calculator import AQICalculator

class AdvancedDataCleaner:
    """Advanced data cleaning and preprocessing"""
    
    def __init__(self):
        self.pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
        self.cleaning_log = []
        self.issues_found = []
        
    def log_action(self, action, details):
        """Log cleaning action"""
        self.cleaning_log.append({
            'action': action,
            'details': details,
            'timestamp': datetime.now()
        })
    
    def detect_datetime_column(self, df):
        """Intelligently detect datetime column"""
        datetime_patterns = ['date', 'time', 'timestamp', 'dt', 'datetime']
        
        # Check column names
        for col in df.columns:
            if any(pattern in col.lower() for pattern in datetime_patterns):
                return col
        
        # Check first column if it looks like datetime
        first_col = df.iloc[:, 0]
        try:
            pd.to_datetime(first_col, errors='coerce')
            if first_col.notna().sum() / len(first_col) > 0.8:
                return df.columns[0]
        except:
            pass
        
        return None
    
    def parse_datetime(self, df, datetime_col):
        """Parse datetime with multiple format attempts"""
        if datetime_col is None:
            self.issues_found.append("❌ No datetime column found")
            return None
        
        # Try multiple datetime formats
        formats = [
            None,  # Auto-detect
            '%Y-%m-%d %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%m/%d/%Y %H:%M',
            '%Y-%m-%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%m/%d/%Y'
        ]
        
        for fmt in formats:
            try:
                if fmt is None:
                    parsed = pd.to_datetime(df[datetime_col], errors='coerce')
                else:
                    parsed = pd.to_datetime(df[datetime_col], format=fmt, errors='coerce')
                
                # Check if parsing was successful for most values
                if parsed.notna().sum() / len(parsed) > 0.8:
                    self.log_action("Datetime Parsing", f"Successfully parsed using format: {fmt or 'auto'}")
                    return parsed
            except:
                continue
        
        self.issues_found.append(f"⚠️ Could not parse datetime column '{datetime_col}'")
        return None
    
    def detect_pollutant_columns(self, df):
        """Detect and map pollutant columns with fuzzy matching"""
        pollutant_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '').replace('.', '')
            
            # Direct matches
            if 'pm2.5' in col_lower or 'pm25' in col_lower:
                pollutant_mapping[col] = 'PM2.5'
            elif 'pm10' in col_lower:
                pollutant_mapping[col] = 'PM10'
            elif 'no2' in col_lower:
                pollutant_mapping[col] = 'NO2'
            elif 'so2' in col_lower:
                pollutant_mapping[col] = 'SO2'
            elif col_lower in ['co', 'carbonmonoxide']:
                pollutant_mapping[col] = 'CO'
            elif 'o3' in col_lower or 'ozone' in col_lower:
                pollutant_mapping[col] = 'O3'
        
        if pollutant_mapping:
            self.log_action("Pollutant Detection", f"Found {len(pollutant_mapping)} pollutant columns")
        else:
            self.issues_found.append("❌ No pollutant columns detected")
        
        return pollutant_mapping
    
    def handle_missing_values(self, df, method='smart'):
        """Handle missing values with smart strategy"""
        initial_missing = df.isnull().sum().sum()
        
        if initial_missing == 0:
            return df
        
        df_clean = df.copy()
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                missing_pct = (df[col].isnull().sum() / len(df)) * 100
                
                if missing_pct > 50:
                    self.issues_found.append(f"⚠️ {col}: {missing_pct:.1f}% missing (high)")
                    # Fill with median for highly missing data
                    df_clean[col].fillna(df[col].median(), inplace=True)
                elif missing_pct > 0:
                    # Interpolate for time series data
                    df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
                    # Fill remaining with forward/backward fill
                    df_clean[col].fillna(method='ffill', inplace=True)
                    df_clean[col].fillna(method='bfill', inplace=True)
        
        final_missing = df_clean.isnull().sum().sum()
        filled = initial_missing - final_missing
        
        self.log_action("Missing Values", f"Filled {filled} missing values ({initial_missing} → {final_missing})")
        
        return df_clean
    
    def remove_outliers(self, df, method='iqr', threshold=3):
        """Remove or cap outliers"""
        df_clean = df.copy()
        outliers_found = {}
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - threshold * IQR
                    upper = Q3 + threshold * IQR
                elif method == 'zscore':
                    mean = df[col].mean()
                    std = df[col].std()
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                
                # Count outliers
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                
                if outliers > 0:
                    outliers_found[col] = outliers
                    # Cap outliers instead of removing
                    df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        
        if outliers_found:
            total_outliers = sum(outliers_found.values())
            self.log_action("Outlier Handling", f"Capped {total_outliers} outliers across {len(outliers_found)} columns")
        
        return df_clean, outliers_found
    
    def remove_duplicates(self, df, datetime_col):
        """Remove duplicate timestamps"""
        if datetime_col in df.columns:
            initial_len = len(df)
            df_clean = df.drop_duplicates(subset=[datetime_col], keep='first')
            duplicates = initial_len - len(df_clean)
            
            if duplicates > 0:
                self.log_action("Duplicate Removal", f"Removed {duplicates} duplicate timestamps")
            
            return df_clean
        
        return df
    
    def ensure_regular_intervals(self, df, target_freq='H'):
        """Ensure regular time intervals"""
        if not isinstance(df.index, pd.DatetimeIndex):
            return df
        
        # Detect current frequency
        time_diffs = df.index.to_series().diff()
        median_diff = time_diffs.median()
        
        # Create regular time range
        start = df.index.min()
        end = df.index.max()
        regular_index = pd.date_range(start=start, end=end, freq=target_freq)
        
        # Reindex and interpolate
        df_regular = df.reindex(regular_index)
        
        # Interpolate missing values
        for col in df_regular.columns:
            if df_regular[col].dtype in ['float64', 'int64']:
                df_regular[col] = df_regular[col].interpolate(method='time')
        
        added_rows = len(df_regular) - len(df)
        
        if added_rows > 0:
            self.log_action("Regular Intervals", f"Created regular {target_freq} intervals, added {added_rows} interpolated rows")
        
        return df_regular
    
    def convert_units(self, df, pollutant_mapping):
        """Convert units if needed"""
        for col, pollutant in pollutant_mapping.items():
            # Check if values are in wrong unit (e.g., ppb instead of μg/m³)
            max_val = df[col].max()
            
            # If values are very small, might be in different unit
            if max_val < 1 and pollutant == 'CO':
                # Convert mg/m³ to ppm or similar
                pass
        
        return df
    
    def validate_ranges(self, df, pollutant_mapping):
        """Validate pollutant ranges"""
        issues = []
        
        # Expected reasonable ranges (μg/m³)
        ranges = {
            'PM2.5': (0, 500),
            'PM10': (0, 1000),
            'NO2': (0, 500),
            'SO2': (0, 500),
            'CO': (0, 50),
            'O3': (0, 500)
        }
        
        for col, pollutant in pollutant_mapping.items():
            if pollutant in ranges:
                min_val, max_val = ranges[pollutant]
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                
                if out_of_range > 0:
                    issues.append(f"⚠️ {pollutant}: {out_of_range} values outside normal range")
        
        return issues
    
    def clean_pipeline(self, df):
        """Complete cleaning pipeline"""
        self.cleaning_log = []
        self.issues_found = []
        
        # Step 1: Detect datetime
        datetime_col = self.detect_datetime_column(df)
        
        # Step 2: Parse datetime
        if datetime_col:
            parsed_dt = self.parse_datetime(df, datetime_col)
            if parsed_dt is not None:
                df['datetime_parsed'] = parsed_dt
                df = df[df['datetime_parsed'].notna()].copy()
                df.set_index('datetime_parsed', inplace=True)
                df.index.name = 'datetime'
        
        # Step 3: Detect pollutants
        pollutant_mapping = self.detect_pollutant_columns(df)
        
        # Step 4: Rename pollutant columns
        df = df.rename(columns=pollutant_mapping)
        
        # Step 5: Keep only relevant columns
        keep_cols = [col for col in df.columns if col in self.pollutants or col in ['city', 'location', 'station']]
        df = df[keep_cols]
        
        # Step 6: Remove duplicates
        if datetime_col:
            df = self.remove_duplicates(df.reset_index(), 'datetime').set_index('datetime')
        
        # Step 7: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 8: Remove outliers
        df, outliers = self.remove_outliers(df)
        
        # Step 9: Ensure regular intervals
        df = self.ensure_regular_intervals(df)
        
        # Step 10: Validate ranges
        range_issues = self.validate_ranges(df, {col: col for col in df.columns if col in self.pollutants})
        self.issues_found.extend(range_issues)
        
        return df, pollutant_mapping


def show_advanced_processor_page():
    """Main page for advanced data processing"""
    st.header("🔧 Advanced Data Cleaning & Processing")
    
    st.markdown("""
    Upload your **raw, messy CSV file** and let our system automatically:
    - 🔍 Detect datetime and pollutant columns
    - 🧹 Clean missing values and outliers
    - 📊 Create regular time intervals
    - ✅ Validate data quality
    - 📈 Generate comprehensive visualizations
    """)
    
    # File uploader
    st.subheader("📁 Upload Raw CSV File")
    
    uploaded_file = st.file_uploader(
        "Drop your CSV file here - any format, any structure",
        type=['csv'],
        help="System will automatically detect and clean your data"
    )
    
    # Processing options
    with st.expander("⚙️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            missing_method = st.selectbox(
                "Missing Value Strategy",
                ["Smart (Interpolation)", "Forward Fill", "Median Fill"],
                help="How to handle missing data"
            )
            
            time_frequency = st.selectbox(
                "Target Time Frequency",
                ["H (Hourly)", "D (Daily)", "30min (Half-hourly)"],
                help="Regularize data to this frequency"
            )
        
        with col2:
            outlier_method = st.selectbox(
                "Outlier Detection",
                ["IQR (Recommended)", "Z-Score", "None"],
                help="Method for detecting outliers"
            )
            
            outlier_threshold = st.slider(
                "Outlier Threshold",
                1.5, 5.0, 3.0, 0.5,
                help="Higher = more lenient"
            )
    
    # Show example of messy data
    with st.expander("📋 Example: What Kind of Messy Data We Can Handle"):
        st.markdown("""
        **We can handle files with:**
        - ❌ Missing datetime column names → We auto-detect
        - ❌ Various date formats → We parse automatically
        - ❌ Misspelled pollutant names → We fuzzy match
        - ❌ Missing values → We interpolate smartly
        - ❌ Outliers → We detect and clean
        - ❌ Irregular time intervals → We regularize
        - ❌ Duplicate timestamps → We remove
        - ❌ Extra columns → We keep only relevant ones
        """)
    
    if uploaded_file is not None:
        try:
            # Read raw CSV
            with st.spinner("Reading raw data..."):
                df_raw = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df_raw)} rows with {len(df_raw.columns)} columns")
            
            # Show raw data preview
            st.subheader("📄 Step 1: Raw Data Preview")
            st.dataframe(df_raw.head(10), use_container_width=True)
            
            # Show raw data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df_raw):,}")
            with col2:
                st.metric("Total Columns", len(df_raw.columns))
            with col3:
                missing_pct = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100
                st.metric("Missing Data", f"{missing_pct:.1f}%")
            
            # Process data
            st.subheader("🔧 Step 2: Data Cleaning & Processing")
            
            with st.spinner("Cleaning and processing data... This may take a moment."):
                cleaner = AdvancedDataCleaner()
                
                # Clean data
                df_clean, pollutant_mapping = cleaner.clean_pipeline(df_raw)
            
            # Show cleaning log
            st.success("✅ Data cleaning completed!")
            
            with st.expander("📋 View Cleaning Log"):
                if cleaner.cleaning_log:
                    for log in cleaner.cleaning_log:
                        st.info(f"**{log['action']}**: {log['details']}")
                else:
                    st.write("No major cleaning actions needed - data was already clean!")
            
            # Show issues found
            if cleaner.issues_found:
                with st.expander("⚠️ Issues Detected & Fixed"):
                    for issue in cleaner.issues_found:
                        st.warning(issue)
            
            # Show cleaned data
            st.subheader("✨ Step 3: Cleaned Data")
            st.dataframe(df_clean.head(20), use_container_width=True)
            
            # Cleaning summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Cleaned Rows", f"{len(df_clean):,}", 
                         delta=f"{len(df_clean) - len(df_raw):+,}")
            with col2:
                pollutants_found = len([col for col in df_clean.columns if col in cleaner.pollutants])
                st.metric("Pollutants Found", pollutants_found)
            with col3:
                missing_after = (df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100
                st.metric("Missing Data", f"{missing_after:.1f}%", 
                         delta=f"{missing_after - missing_pct:.1f}%")
            with col4:
                if isinstance(df_clean.index, pd.DatetimeIndex):
                    time_span = (df_clean.index[-1] - df_clean.index[0]).days
                    st.metric("Time Span", f"{time_span} days")
            
            # Show comparison
            st.subheader("📊 Step 4: Before vs After Comparison")
            
            show_before_after_comparison(df_raw, df_clean, pollutant_mapping)
            
            # Show comprehensive analysis
            st.subheader("📈 Step 5: Comprehensive Data Analysis")
            
            show_comprehensive_analysis(df_clean)
            
            # Download options
            st.subheader("💾 Step 6: Download Processed Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Download cleaned CSV
                csv = df_clean.to_csv()
                st.download_button(
                    label="📥 Download Cleaned Data (CSV)",
                    data=csv,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Download cleaning report
                report = generate_cleaning_report(df_raw, df_clean, cleaner)
                st.download_button(
                    label="📥 Download Cleaning Report (TXT)",
                    data=report,
                    file_name=f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                # Download statistics
                stats = generate_statistics_json(df_clean)
                st.download_button(
                    label="📥 Download Statistics (JSON)",
                    data=stats,
                    file_name=f"statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your file is a valid CSV format")
            
            with st.expander("🔍 Show Error Details"):
                st.code(str(e))


def show_before_after_comparison(df_raw, df_clean, pollutant_mapping):
    """Show before/after comparison"""
    
    tabs = st.tabs(["📊 Data Quality", "📈 Missing Values", "🎯 Outliers", "📅 Time Coverage"])
    
    # Tab 1: Data Quality
    with tabs[0]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Before Cleaning")
            before_stats = {
                'Total Rows': len(df_raw),
                'Total Columns': len(df_raw.columns),
                'Missing Values': df_raw.isnull().sum().sum(),
                'Missing %': f"{(df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100:.1f}%"
            }
            st.json(before_stats)
        
        with col2:
            st.markdown("### After Cleaning")
            after_stats = {
                'Total Rows': len(df_clean),
                'Total Columns': len(df_clean.columns),
                'Missing Values': df_clean.isnull().sum().sum(),
                'Missing %': f"{(df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100:.1f}%"
            }
            st.json(after_stats)
    
    # Tab 2: Missing Values
    with tabs[1]:
        # Get common numeric columns
        numeric_cols_raw = df_raw.select_dtypes(include=[np.number]).columns
        numeric_cols_clean = df_clean.select_dtypes(include=[np.number]).columns
        common_cols = list(set(numeric_cols_raw) & set(numeric_cols_clean))
        
        if common_cols:
            missing_before = []
            missing_after = []
            
            for col in common_cols[:10]:  # Limit to 10 columns
                missing_before.append({
                    'Column': col,
                    'Missing': df_raw[col].isnull().sum(),
                    'Percentage': f"{(df_raw[col].isnull().sum() / len(df_raw)) * 100:.1f}%"
                })
                
                clean_col = pollutant_mapping.get(col, col)
                if clean_col in df_clean.columns:
                    missing_after.append({
                        'Column': clean_col,
                        'Missing': df_clean[clean_col].isnull().sum(),
                        'Percentage': f"{(df_clean[clean_col].isnull().sum() / len(df_clean)) * 100:.1f}%"
                    })
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Before")
                st.dataframe(pd.DataFrame(missing_before), use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### After")
                st.dataframe(pd.DataFrame(missing_after), use_container_width=True, hide_index=True)
    
    # Tab 3: Outliers
    with tabs[2]:
        if len(common_cols) > 0:
            selected_col = st.selectbox("Select Column to Compare", common_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Before Cleaning")
                fig = px.box(df_raw, y=selected_col, title=f"{selected_col} - Raw Data")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### After Cleaning")
                clean_col = pollutant_mapping.get(selected_col, selected_col)
                if clean_col in df_clean.columns:
                    fig = px.box(df_clean, y=clean_col, title=f"{clean_col} - Cleaned Data")
                    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Time Coverage
    with tabs[3]:
        if isinstance(df_clean.index, pd.DatetimeIndex):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Before")
                st.write(f"- **Start**: {df_clean.index.min()}")
                st.write(f"- **End**: {df_clean.index.max()}")
                st.write(f"- **Span**: {(df_clean.index.max() - df_clean.index.min()).days} days")
                st.write(f"- **Data Points**: {len(df_clean)}")
            
            with col2:
                st.markdown("### Time Distribution")
                # Count data points per day
                daily_counts = df_clean.resample('D').size()
                fig = px.bar(x=daily_counts.index, y=daily_counts.values,
                           title="Data Points per Day",
                           labels={'x': 'Date', 'y': 'Count'})
                st.plotly_chart(fig, use_container_width=True)


def show_comprehensive_analysis(df):
    """Show comprehensive analysis of cleaned data"""
    
    pollutants = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
    
    if not pollutants:
        st.warning("No pollutant columns found in cleaned data")
        return
    
    tabs = st.tabs([
        "📊 Summary Statistics",
        "📈 Time Series",
        "📉 Distributions",
        "🔗 Correlations",
        "📅 Patterns",
        "🎯 Quality Metrics"
    ])
    
    # Tab 1: Summary Statistics
    with tabs[0]:
        st.markdown("### Statistical Summary")
        
        stats_df = df[pollutants].describe().T
        stats_df['Missing %'] = [(df[col].isnull().sum() / len(df)) * 100 for col in pollutants]
        
        # Format and display
        styled_stats = stats_df.style.format({
            'count': '{:.0f}',
            'mean': '{:.2f}',
            'std': '{:.2f}',
            'min': '{:.2f}',
            '25%': '{:.2f}',
            '50%': '{:.2f}',
            '75%': '{:.2f}',
            'max': '{:.2f}',
            'Missing %': '{:.2f}'
        })
        
        st.dataframe(styled_stats, use_container_width=True)
        
        # Quick insights
        st.markdown("### 🔍 Quick Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Highest average
            highest = df[pollutants].mean().idxmax()
            highest_val = df[pollutants].mean().max()
            st.metric(f"Highest Average Pollutant", highest, f"{highest_val:.2f} μg/m³")
            
            # Most variable
            most_variable = df[pollutants].std().idxmax()
            var_val = df[pollutants].std().max()
            st.metric("Most Variable Pollutant", most_variable, f"σ = {var_val:.2f}")
        
        with col2:
            # Data completeness
            completeness = (1 - df[pollutants].isnull().sum().sum() / (len(df) * len(pollutants))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
            
            # Time span
            if isinstance(df.index, pd.DatetimeIndex):
                span = (df.index[-1] - df.index[0]).days
                st.metric("Time Span", f"{span} days")
    
    # Tab 2: Time Series
    with tabs[1]:
        st.markdown("### Pollutant Trends Over Time")
        
        selected_pollutants = st.multiselect(
            "Select Pollutants to Display",
            pollutants,
            default=[pollutants[0]] if pollutants else []
        )
        
        if selected_pollutants:
            fig = go.Figure()
            
            for pollutant in selected_pollutants:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[pollutant],
                    name=pollutant,
                    mode='lines'
                ))
            
            fig.update_layout(
                title="Time Series Comparison",
                xaxis_title="Date",
                yaxis_title="Concentration (μg/m³)",
                hovermode='x unified',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent data
            st.markdown("### 📋 Recent Data (Last 24 hours)")
            recent_data = df[selected_pollutants].tail(24)
            st.dataframe(recent_data, use_container_width=True)
    
    # Tab 3: Distributions
    with tabs[2]:
        st.markdown("### Distribution Analysis")
        
        selected_pollutant = st.selectbox("Select Pollutant", pollutants, key="dist_select")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig = px.histogram(df, x=selected_pollutant, nbins=50,
                             title=f"{selected_pollutant} Distribution")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Percentile info
            st.markdown("**Percentiles:**")
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            perc_data = []
            for p in percentiles:
                val = df[selected_pollutant].quantile(p/100)
                perc_data.append({'Percentile': f'{p}th', 'Value': f'{val:.2f}'})
            
            st.dataframe(pd.DataFrame(perc_data), use_container_width=True, hide_index=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_pollutant,
                        title=f"{selected_pollutant} Box Plot")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistical summary
            st.markdown("**Statistics:**")
            stats = {
                'Mean': f"{df[selected_pollutant].mean():.2f}",
                'Median': f"{df[selected_pollutant].median():.2f}",
                'Std Dev': f"{df[selected_pollutant].std():.2f}",
                'Skewness': f"{df[selected_pollutant].skew():.2f}",
                'Min': f"{df[selected_pollutant].min():.2f}",
                'Max': f"{df[selected_pollutant].max():.2f}"
            }
            for k, v in stats.items():
                st.write(f"**{k}:** {v}")
    
    # Tab 4: Correlations
    with tabs[3]:
        st.markdown("### Correlation Analysis")
        
        if len(pollutants) > 1:
            # Correlation matrix
            corr = df[pollutants].corr()
            
            fig = px.imshow(corr,
                           labels=dict(color="Correlation"),
                           x=pollutants,
                           y=pollutants,
                           color_continuous_scale='RdBu_r',
                           aspect="auto",
                           zmin=-1, zmax=1)
            fig.update_layout(
                title='Pollutant Correlation Matrix',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            st.markdown("### 🔗 Strong Correlations")
            
            strong_corr = []
            for i, poll1 in enumerate(pollutants):
                for j, poll2 in enumerate(pollutants):
                    if i < j:
                        corr_val = corr.loc[poll1, poll2]
                        if abs(corr_val) > 0.5:
                            strength = "Very Strong" if abs(corr_val) > 0.8 else "Strong" if abs(corr_val) > 0.6 else "Moderate"
                            direction = "Positive" if corr_val > 0 else "Negative"
                            strong_corr.append({
                                'Pollutant 1': poll1,
                                'Pollutant 2': poll2,
                                'Correlation': f"{corr_val:.3f}",
                                'Strength': strength,
                                'Direction': direction
                            })
            
            if strong_corr:
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True, hide_index=True)
            else:
                st.info("No strong correlations found between pollutants")
        else:
            st.info("Need at least 2 pollutants for correlation analysis")
    
    # Tab 5: Patterns
    with tabs[4]:
        st.markdown("### Temporal Patterns")
        
        if isinstance(df.index, pd.DatetimeIndex):
            pattern_pollutant = st.selectbox("Select Pollutant", pollutants, key="pattern_select")
            
            # Add temporal features
            df_temp = df.copy()
            df_temp['hour'] = df_temp.index.hour
            df_temp['day_of_week'] = df_temp.index.day_name()
            df_temp['month'] = df_temp.index.month_name()
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Hourly pattern
                st.markdown("**⏰ Hourly Pattern**")
                hourly = df_temp.groupby('hour')[pattern_pollutant].agg(['mean', 'std'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hourly.index,
                    y=hourly['mean'],
                    mode='lines+markers',
                    name='Mean',
                    line=dict(color='blue', width=2)
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=list(hourly.index) + list(hourly.index[::-1]),
                    y=list(hourly['mean'] + hourly['std']) + list((hourly['mean'] - hourly['std'])[::-1]),
                    fill='toself',
                    fillcolor='rgba(0,100,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Std Dev',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"{pattern_pollutant} - Hourly Pattern",
                    xaxis_title="Hour of Day",
                    yaxis_title=f"{pattern_pollutant} (μg/m³)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Peak hours
                peak_hour = hourly['mean'].idxmax()
                low_hour = hourly['mean'].idxmin()
                st.info(f"**Peak:** {peak_hour}:00 ({hourly['mean'][peak_hour]:.2f})\n\n"
                       f"**Lowest:** {low_hour}:00 ({hourly['mean'][low_hour]:.2f})")
            
            with col2:
                # Day of week pattern
                st.markdown("**📅 Day of Week Pattern**")
                dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow = df_temp.groupby('day_of_week')[pattern_pollutant].mean().reindex(dow_order)
                
                fig = px.bar(x=dow.index, y=dow.values,
                           title=f"{pattern_pollutant} - Weekly Pattern")
                fig.update_layout(
                    xaxis_title="Day of Week",
                    yaxis_title=f"{pattern_pollutant} (μg/m³)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekday vs weekend
                df_temp['is_weekend'] = df_temp.index.dayofweek >= 5
                weekday_avg = df_temp[~df_temp['is_weekend']][pattern_pollutant].mean()
                weekend_avg = df_temp[df_temp['is_weekend']][pattern_pollutant].mean()
                diff_pct = ((weekend_avg - weekday_avg) / weekday_avg) * 100
                
                st.info(f"**Weekday Avg:** {weekday_avg:.2f}\n\n"
                       f"**Weekend Avg:** {weekend_avg:.2f}\n\n"
                       f"**Difference:** {diff_pct:+.1f}%")
            
            # Monthly pattern (if enough data)
            if len(df_temp['month'].unique()) > 3:
                st.markdown("**📆 Monthly Pattern**")
                month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                              'July', 'August', 'September', 'October', 'November', 'December']
                monthly = df_temp.groupby('month')[pattern_pollutant].mean()
                monthly = monthly.reindex([m for m in month_order if m in monthly.index])
                
                fig = px.line(x=monthly.index, y=monthly.values,
                            title=f"{pattern_pollutant} - Monthly Pattern",
                            markers=True)
                fig.update_layout(
                    xaxis_title="Month",
                    yaxis_title=f"{pattern_pollutant} (μg/m³)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Temporal analysis requires datetime index")
    
    # Tab 6: Quality Metrics
    with tabs[5]:
        st.markdown("### Data Quality Assessment")
        
        quality_scores = []
        
        for pollutant in pollutants:
            # Calculate quality metrics
            missing_pct = (df[pollutant].isnull().sum() / len(df)) * 100
            
            # Outlier detection
            Q1 = df[pollutant].quantile(0.25)
            Q3 = df[pollutant].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[pollutant] < Q1 - 3*IQR) | (df[pollutant] > Q3 + 3*IQR)).sum()
            outlier_pct = (outliers / len(df)) * 100
            
            # Variability (coefficient of variation)
            cv = (df[pollutant].std() / df[pollutant].mean()) * 100 if df[pollutant].mean() > 0 else 0
            
            # Overall quality score (0-100)
            completeness_score = 100 - missing_pct
            outlier_score = max(0, 100 - outlier_pct * 5)
            consistency_score = max(0, 100 - min(cv, 100))
            
            overall_score = (completeness_score * 0.5 + outlier_score * 0.3 + consistency_score * 0.2)
            
            quality_scores.append({
                'Pollutant': pollutant,
                'Completeness': f"{completeness_score:.1f}%",
                'Outliers': f"{outlier_pct:.1f}%",
                'Variability (CV)': f"{cv:.1f}%",
                'Quality Score': f"{overall_score:.1f}/100"
            })
        
        # Display quality table
        quality_df = pd.DataFrame(quality_scores)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)
        
        # Visual quality scores
        st.markdown("### 📊 Quality Scores Visualization")
        
        scores_only = [float(q['Quality Score'].split('/')[0]) for q in quality_scores]
        
        fig = go.Figure(go.Bar(
            x=[q['Pollutant'] for q in quality_scores],
            y=scores_only,
            marker=dict(
                color=scores_only,
                colorscale='RdYlGn',
                cmin=0,
                cmax=100,
                colorbar=dict(title="Quality Score")
            ),
            text=[f"{s:.1f}" for s in scores_only],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Data Quality Scores by Pollutant",
            xaxis_title="Pollutant",
            yaxis_title="Quality Score (0-100)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Overall assessment
        avg_score = np.mean(scores_only)
        
        if avg_score >= 90:
            st.success(f"🎉 **Excellent Data Quality** (Average Score: {avg_score:.1f}/100)\n\nData is highly reliable for analysis and predictions.")
        elif avg_score >= 75:
            st.info(f"✅ **Good Data Quality** (Average Score: {avg_score:.1f}/100)\n\nData is suitable for most analyses with minor limitations.")
        elif avg_score >= 60:
            st.warning(f"⚠️ **Fair Data Quality** (Average Score: {avg_score:.1f}/100)\n\nData can be used but results should be interpreted with caution.")
        else:
            st.error(f"❌ **Poor Data Quality** (Average Score: {avg_score:.1f}/100)\n\nSignificant data quality issues detected. Consider additional cleaning or data collection.")


def generate_cleaning_report(df_raw, df_clean, cleaner):
    """Generate comprehensive cleaning report"""
    report = []
    
    report.append("=" * 80)
    report.append("DATA CLEANING & PREPROCESSING REPORT")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Original data info
    report.append("=" * 80)
    report.append("1. ORIGINAL DATA")
    report.append("=" * 80)
    report.append(f"Total Rows: {len(df_raw):,}")
    report.append(f"Total Columns: {len(df_raw.columns)}")
    report.append(f"Missing Values: {df_raw.isnull().sum().sum():,}")
    report.append(f"Missing Percentage: {(df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100:.2f}%")
    
    # Cleaned data info
    report.append("\n" + "=" * 80)
    report.append("2. CLEANED DATA")
    report.append("=" * 80)
    report.append(f"Total Rows: {len(df_clean):,}")
    report.append(f"Total Columns: {len(df_clean.columns)}")
    report.append(f"Missing Values: {df_clean.isnull().sum().sum():,}")
    report.append(f"Missing Percentage: {(df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100:.2f}%")
    
    if isinstance(df_clean.index, pd.DatetimeIndex):
        report.append(f"\nTime Range:")
        report.append(f"  Start: {df_clean.index.min()}")
        report.append(f"  End: {df_clean.index.max()}")
        report.append(f"  Span: {(df_clean.index.max() - df_clean.index.min()).days} days")
    
    # Cleaning actions
    report.append("\n" + "=" * 80)
    report.append("3. CLEANING ACTIONS PERFORMED")
    report.append("=" * 80)
    
    if cleaner.cleaning_log:
        for i, log in enumerate(cleaner.cleaning_log, 1):
            report.append(f"\n{i}. {log['action']}")
            report.append(f"   {log['details']}")
    else:
        report.append("No major cleaning actions needed.")
    
    # Issues found
    if cleaner.issues_found:
        report.append("\n" + "=" * 80)
        report.append("4. ISSUES DETECTED & RESOLVED")
        report.append("=" * 80)
        for i, issue in enumerate(cleaner.issues_found, 1):
            report.append(f"\n{i}. {issue}")
    
    # Summary
    report.append("\n" + "=" * 80)
    report.append("5. SUMMARY")
    report.append("=" * 80)
    
    rows_change = len(df_clean) - len(df_raw)
    report.append(f"Rows Changed: {rows_change:+,}")
    report.append(f"Columns Retained: {len(df_clean.columns)}")
    
    missing_before = (df_raw.isnull().sum().sum() / (len(df_raw) * len(df_raw.columns))) * 100
    missing_after = (df_clean.isnull().sum().sum() / (len(df_clean) * len(df_clean.columns))) * 100
    report.append(f"Missing Data Reduction: {missing_before:.2f}% → {missing_after:.2f}%")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return "\n".join(report)


def generate_statistics_json(df):
    """Generate statistics in JSON format"""
    
    pollutants = [col for col in df.columns if col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']]
    
    stats = {
        'metadata': {
            'generated': datetime.now().isoformat(),
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'pollutants': pollutants
        },
        'time_range': {},
        'pollutant_statistics': {},
        'quality_metrics': {}
    }
    
    # Time range
    if isinstance(df.index, pd.DatetimeIndex):
        stats['time_range'] = {
            'start': df.index.min().isoformat(),
            'end': df.index.max().isoformat(),
            'span_days': int((df.index.max() - df.index.min()).days)
        }
    
    # Pollutant statistics
    for pollutant in pollutants:
        stats['pollutant_statistics'][pollutant] = {
            'count': int(df[pollutant].count()),
            'mean': float(df[pollutant].mean()),
            'std': float(df[pollutant].std()),
            'min': float(df[pollutant].min()),
            'q25': float(df[pollutant].quantile(0.25)),
            'median': float(df[pollutant].median()),
            'q75': float(df[pollutant].quantile(0.75)),
            'max': float(df[pollutant].max()),
            'missing_count': int(df[pollutant].isnull().sum()),
            'missing_percentage': float((df[pollutant].isnull().sum() / len(df)) * 100)
        }
    
    # Quality metrics
    for pollutant in pollutants:
        missing_pct = (df[pollutant].isnull().sum() / len(df)) * 100
        Q1 = df[pollutant].quantile(0.25)
        Q3 = df[pollutant].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[pollutant] < Q1 - 3*IQR) | (df[pollutant] > Q3 + 3*IQR)).sum()
        
        stats['quality_metrics'][pollutant] = {
            'completeness': float(100 - missing_pct),
            'outlier_count': int(outliers),
            'outlier_percentage': float((outliers / len(df)) * 100),
            'coefficient_of_variation': float((df[pollutant].std() / df[pollutant].mean()) * 100) if df[pollutant].mean() > 0 else 0
        }
    
    return json.dumps(stats, indent=2)


# Main function to integrate with app.py
def show_advanced_data_processor():
    """Main function to call from app.py"""
    show_advanced_processor_page()


if __name__ == "__main__":
    # For standalone testing
    show_advanced_data_processor()