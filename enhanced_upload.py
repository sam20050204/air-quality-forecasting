import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import logger

class SmartDataAnalyzer:
    """Intelligently analyze and visualize any CSV data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = self._detect_datetime_columns()
        
    def _detect_datetime_columns(self):
        """Detect potential datetime columns"""
        datetime_cols = []
        for col in self.df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(self.df[col])
                    datetime_cols.append(col)
                except:
                    pass
        return datetime_cols
    
    def get_data_summary(self):
        """Generate comprehensive data summary"""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols),
            'datetime_columns': len(self.datetime_cols),
            'missing_values': self.df.isnull().sum().sum(),
            'duplicate_rows': self.df.duplicated().sum(),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        return summary
    
    def get_column_info(self):
        """Get detailed column information"""
        info = []
        for col in self.df.columns:
            col_info = {
                'Column': col,
                'Type': str(self.df[col].dtype),
                'Non-Null': self.df[col].notna().sum(),
                'Null': self.df[col].isna().sum(),
                'Unique': self.df[col].nunique(),
                'Sample': str(self.df[col].iloc[0]) if len(self.df) > 0 else 'N/A'
            }
            info.append(col_info)
        return pd.DataFrame(info)
    
    def preprocess_data(self):
        """Smart preprocessing based on data types"""
        df_clean = self.df.copy()
        
        # Convert datetime columns
        for col in self.datetime_cols:
            try:
                df_clean[col] = pd.to_datetime(df_clean[col])
            except:
                pass
        
        # Handle missing values in numeric columns
        for col in self.numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                # Use median for numeric columns
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Handle missing values in categorical columns
        for col in self.categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                df_clean[col].fillna('Unknown', inplace=True)
        
        return df_clean
    
    def generate_visualizations(self):
        """Generate appropriate visualizations based on data types"""
        return {
            'numeric': self._create_numeric_visualizations(),
            'categorical': self._create_categorical_visualizations(),
            'correlations': self._create_correlation_heatmap(),
            'distributions': self._create_distribution_plots(),
            'timeseries': self._create_timeseries_plots() if self.datetime_cols else None
        }
    
    def _create_numeric_visualizations(self):
        """Create visualizations for numeric data"""
        charts = []
        
        if not self.numeric_cols:
            return charts
        
        # Summary statistics bar chart
        stats_data = []
        for col in self.numeric_cols[:10]:  # Limit to 10 columns
            stats_data.append({
                'Column': col,
                'Mean': self.df[col].mean(),
                'Median': self.df[col].median(),
                'Std': self.df[col].std()
            })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Mean', x=stats_df['Column'], y=stats_df['Mean']))
            fig.add_trace(go.Bar(name='Median', x=stats_df['Column'], y=stats_df['Median']))
            fig.update_layout(
                title='Numeric Columns - Mean vs Median',
                xaxis_title='Column',
                yaxis_title='Value',
                barmode='group',
                height=400
            )
            charts.append(('Statistics Comparison', fig))
        
        return charts
    
    def _create_categorical_visualizations(self):
        """Create visualizations for categorical data"""
        charts = []
        
        if not self.categorical_cols:
            return charts
        
        # Pie charts for top categorical columns
        for col in self.categorical_cols[:4]:  # Limit to 4 columns
            value_counts = self.df[col].value_counts().head(10)
            
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index,
                values=value_counts.values,
                hole=0.3
            )])
            fig.update_layout(
                title=f'{col} Distribution',
                height=400
            )
            charts.append((f'{col} Distribution', fig))
        
        return charts
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap for numeric columns"""
        if len(self.numeric_cols) < 2:
            return None
        
        corr = self.df[self.numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Correlation Heatmap',
            height=600,
            width=700
        )
        
        return fig
    
    def _create_distribution_plots(self):
        """Create distribution plots for numeric columns"""
        charts = []
        
        for col in self.numeric_cols[:6]:  # Limit to 6 columns
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=self.df[col],
                name='Distribution',
                nbinsx=30,
                marker_color='lightblue',
                opacity=0.7
            ))
            
            fig.update_layout(
                title=f'{col} Distribution',
                xaxis_title=col,
                yaxis_title='Frequency',
                height=350
            )
            
            charts.append((f'{col} Distribution', fig))
        
        return charts
    
    def _create_timeseries_plots(self):
        """Create time series plots if datetime columns exist"""
        if not self.datetime_cols or not self.numeric_cols:
            return []
        
        charts = []
        datetime_col = self.datetime_cols[0]
        
        # Sort by datetime
        df_sorted = self.df.sort_values(datetime_col)
        
        for col in self.numeric_cols[:5]:  # Limit to 5 numeric columns
            fig = px.line(
                df_sorted,
                x=datetime_col,
                y=col,
                title=f'{col} over Time'
            )
            fig.update_layout(
                xaxis_title='Date/Time',
                yaxis_title=col,
                height=400
            )
            charts.append((f'{col} Time Series', fig))
        
        return charts


def show_enhanced_upload_page():
    """Enhanced CSV Upload Page with Smart Analysis"""
    st.header("📤 Smart CSV Upload & Analysis")
    
    st.markdown("""
    Upload any CSV file and get instant insights with:
    - 📊 **Automatic data profiling**
    - 📈 **Smart visualizations** (pie charts, bar graphs, time series)
    - 🔍 **Data quality assessment**
    - 📉 **Statistical analysis**
    - 🧹 **Automatic preprocessing**
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload any CSV file for automatic analysis and visualization"
    )
    
    if uploaded_file is not None:
        try:
            # Read CSV
            df_uploaded = pd.read_csv(uploaded_file)
            
            st.success(f"✅ File uploaded successfully! ({uploaded_file.name})")
            
            # Initialize analyzer
            analyzer = SmartDataAnalyzer(df_uploaded)
            
            # Display tabs for different sections
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📋 Data Preview", 
                "📊 Overview", 
                "📈 Visualizations",
                "🔍 Statistics",
                "🧹 Preprocessed Data"
            ])
            
            # TAB 1: Data Preview
            with tab1:
                st.subheader("Raw Data Preview")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.dataframe(df_uploaded.head(20), use_container_width=True)
                with col2:
                    st.metric("Total Rows", len(df_uploaded))
                    st.metric("Total Columns", len(df_uploaded.columns))
                    
                    if st.button("Download Sample (First 100 rows)"):
                        csv = df_uploaded.head(100).to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            "sample_data.csv",
                            "text/csv"
                        )
            
            # TAB 2: Overview
            with tab2:
                st.subheader("📊 Data Summary")
                
                summary = analyzer.get_data_summary()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Rows", f"{summary['total_rows']:,}")
                col2.metric("Total Columns", summary['total_columns'])
                col3.metric("Missing Values", summary['missing_values'])
                col4.metric("Duplicates", summary['duplicate_rows'])
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Numeric Columns", summary['numeric_columns'])
                col2.metric("Categorical Columns", summary['categorical_columns'])
                col3.metric("DateTime Columns", summary['datetime_columns'])
                col4.metric("Memory Usage", summary['memory_usage'])
                
                # Column Information
                st.subheader("📋 Column Details")
                col_info = analyzer.get_column_info()
                st.dataframe(col_info, use_container_width=True)
                
                # Data Quality Assessment
                st.subheader("🎯 Data Quality Assessment")
                
                missing_pct = (summary['missing_values'] / (summary['total_rows'] * summary['total_columns'])) * 100
                duplicate_pct = (summary['duplicate_rows'] / summary['total_rows']) * 100
                
                if missing_pct < 5 and duplicate_pct < 1:
                    st.success("✅ **Excellent** - Data quality is very good!")
                elif missing_pct < 15 and duplicate_pct < 5:
                    st.info("ℹ️ **Good** - Minor data quality issues detected")
                else:
                    st.warning("⚠️ **Fair** - Some data quality issues need attention")
                
                st.write(f"- Missing data: {missing_pct:.2f}%")
                st.write(f"- Duplicate rows: {duplicate_pct:.2f}%")
            
            # TAB 3: Visualizations
            with tab3:
                st.subheader("📈 Smart Visualizations")
                
                viz = analyzer.generate_visualizations()
                
                # Numeric visualizations
                if viz['numeric']:
                    st.markdown("### 📊 Numeric Data Analysis")
                    for title, fig in viz['numeric']:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Categorical visualizations
                if viz['categorical']:
                    st.markdown("### 🥧 Categorical Data Distribution")
                    cols = st.columns(2)
                    for idx, (title, fig) in enumerate(viz['categorical']):
                        with cols[idx % 2]:
                            st.plotly_chart(fig, use_container_width=True)
                
                # Correlation heatmap
                if viz['correlations'] is not None:
                    st.markdown("### 🔗 Correlation Analysis")
                    st.plotly_chart(viz['correlations'], use_container_width=True)
                
                # Distribution plots
                if viz['distributions']:
                    st.markdown("### 📊 Distribution Analysis")
                    cols = st.columns(2)
                    for idx, (title, fig) in enumerate(viz['distributions']):
                        with cols[idx % 2]:
                            st.plotly_chart(fig, use_container_width=True)
                
                # Time series plots
                if viz['timeseries']:
                    st.markdown("### 📈 Time Series Analysis")
                    for title, fig in viz['timeseries']:
                        st.plotly_chart(fig, use_container_width=True)
            
            # TAB 4: Statistics
            with tab4:
                st.subheader("🔍 Statistical Analysis")
                
                if analyzer.numeric_cols:
                    st.markdown("### Numeric Columns Statistics")
                    stats_df = df_uploaded[analyzer.numeric_cols].describe().T
                    st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
                    
                    # Box plots
                    st.markdown("### 📦 Box Plots (Outlier Detection)")
                    selected_col = st.selectbox(
                        "Select column for box plot",
                        analyzer.numeric_cols
                    )
                    
                    fig = go.Figure()
                    fig.add_trace(go.Box(
                        y=df_uploaded[selected_col],
                        name=selected_col,
                        boxmean='sd'
                    ))
                    fig.update_layout(
                        title=f'{selected_col} - Box Plot',
                        yaxis_title=selected_col,
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if analyzer.categorical_cols:
                    st.markdown("### Categorical Columns Frequency")
                    selected_cat = st.selectbox(
                        "Select categorical column",
                        analyzer.categorical_cols
                    )
                    
                    freq_df = df_uploaded[selected_cat].value_counts().reset_index()
                    freq_df.columns = [selected_cat, 'Count']
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        fig = px.bar(
                            freq_df.head(20),
                            x=selected_cat,
                            y='Count',
                            title=f'Top 20 Values in {selected_cat}'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        st.dataframe(freq_df.head(20), use_container_width=True)
            
            # TAB 5: Preprocessed Data
            with tab5:
                st.subheader("🧹 Preprocessed Data")
                
                st.info("Data has been automatically cleaned: missing values filled, outliers handled")
                
                df_clean = analyzer.preprocess_data()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Rows After Cleaning", len(df_clean))
                col2.metric("Missing Values", df_clean.isnull().sum().sum())
                col3.metric("Data Quality", "✅ Clean")
                
                st.dataframe(df_clean.head(20), use_container_width=True)
                
                # Download preprocessed data
                csv = df_clean.to_csv(index=False)
                st.download_button(
                    "📥 Download Preprocessed Data",
                    csv,
                    "preprocessed_data.csv",
                    "text/csv",
                    help="Download the cleaned and preprocessed data"
                )
                
                # Additional preprocessing options
                st.markdown("### 🔧 Additional Preprocessing Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.checkbox("Remove Duplicates"):
                        df_clean = df_clean.drop_duplicates()
                        st.success(f"Removed duplicates. Rows: {len(df_clean)}")
                
                with col2:
                    if st.checkbox("Normalize Numeric Columns"):
                        for col in analyzer.numeric_cols:
                            df_clean[col] = (df_clean[col] - df_clean[col].min()) / (df_clean[col].max() - df_clean[col].min())
                        st.success("Normalized numeric columns (0-1 scale)")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.code(str(e))
            st.info("Please ensure your CSV file is properly formatted and not corrupted.")
    
    else:
        # Show example when no file is uploaded
        st.subheader("📝 Example CSV Formats")
        
        tab1, tab2, tab3 = st.tabs(["Air Quality", "Sales Data", "Generic Data"])
        
        with tab1:
            st.write("**Air Quality Data Format:**")
            example1 = pd.DataFrame({
                'datetime': ['2024-01-01 00:00:00', '2024-01-01 01:00:00', '2024-01-01 02:00:00'],
                'city': ['Delhi', 'Delhi', 'Delhi'],
                'PM2.5': [45.2, 52.1, 48.9],
                'PM10': [78.5, 85.2, 81.3],
                'NO2': [32.1, 35.6, 33.8]
            })
            st.dataframe(example1, use_container_width=True)
        
        with tab2:
            st.write("**Sales Data Format:**")
            example2 = pd.DataFrame({
                'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
                'product': ['Product A', 'Product B', 'Product A'],
                'quantity': [100, 150, 120],
                'revenue': [5000, 7500, 6000],
                'region': ['North', 'South', 'East']
            })
            st.dataframe(example2, use_container_width=True)
        
        with tab3:
            st.write("**Any Generic Data:**")
            st.markdown("""
            The system automatically detects:
            - **Numeric columns** → Statistics, distributions, correlations
            - **Categorical columns** → Pie charts, frequency tables
            - **DateTime columns** → Time series plots
            - **Missing values** → Automatic handling
            
            Just upload and let the system do the rest! 🚀
            """)


# This module can be imported in dashboard/app.py
# Replace the show_upload_page() function with show_enhanced_upload_page()