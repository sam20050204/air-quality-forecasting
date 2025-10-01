import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from src.utils.logger import logger
from src.utils.config import config

class CSVUploader:
    """Handle CSV file uploads and validation"""
    
    def __init__(self):
        self.required_columns = ['datetime']
        self.pollutants = config.data_config['pollutants']
        self.upload_dir = Path('data/uploaded')
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def validate_csv(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate uploaded CSV file structure"""
        errors = []
        
        # Check if dataframe is empty
        if df.empty:
            errors.append("CSV file is empty")
            return False, errors
        
        # Check for datetime column
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if not datetime_cols:
            errors.append("No datetime column found. Expected column with 'date' or 'time' in name")
        
        # Check for at least one pollutant column
        found_pollutants = [p for p in self.pollutants if p in df.columns]
        if not found_pollutants:
            errors.append(f"No pollutant columns found. Expected at least one of: {', '.join(self.pollutants)}")
        
        # Check for minimum number of rows
        if len(df) < 50:
            errors.append(f"Insufficient data. Found {len(df)} rows, need at least 50 for reliable predictions")
        
        # Validate data types
        for pollutant in found_pollutants:
            if not pd.api.types.is_numeric_dtype(df[pollutant]):
                errors.append(f"Column '{pollutant}' must contain numeric values")
        
        return len(errors) == 0, errors
    
    def preprocess_uploaded_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess uploaded CSV data"""
        logger.info("Preprocessing uploaded data")
        
        # Find datetime column
        datetime_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                datetime_col = col
                break
        
        if datetime_col:
            # Convert to datetime
            try:
                df['datetime'] = pd.to_datetime(df[datetime_col])
            except Exception as e:
                logger.warning(f"Could not parse datetime column: {e}")
                # Generate datetime range
                df['datetime'] = pd.date_range(
                    start=datetime.now(),
                    periods=len(df),
                    freq='H'
                )
            
            # Drop original datetime column if different
            if datetime_col != 'datetime' and datetime_col in df.columns:
                df = df.drop(columns=[datetime_col])
        else:
            # Generate datetime if not present
            df['datetime'] = pd.date_range(
                start=datetime.now(),
                periods=len(df),
                freq='H'
            )
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Sort by datetime
        df.sort_index(inplace=True)
        
        # Keep only pollutant columns that exist
        available_pollutants = [p for p in self.pollutants if p in df.columns]
        other_cols = [col for col in df.columns if col not in self.pollutants]
        
        # Keep pollutants and useful metadata
        cols_to_keep = available_pollutants + [col for col in other_cols if col in ['city', 'location', 'station']]
        df = df[cols_to_keep]
        
        # Handle missing values
        for pollutant in available_pollutants:
            # Interpolate missing values
            df[pollutant] = df[pollutant].interpolate(method='linear', limit_direction='both')
            
            # Fill any remaining NaN with median
            if df[pollutant].isnull().any():
                df[pollutant].fillna(df[pollutant].median(), inplace=True)
        
        # Remove outliers (cap at 3 IQR)
        for pollutant in available_pollutants:
            Q1 = df[pollutant].quantile(0.25)
            Q3 = df[pollutant].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df[pollutant] = df[pollutant].clip(lower_bound, upper_bound)
        
        logger.info(f"Preprocessed data: {len(df)} rows, {len(available_pollutants)} pollutants")
        
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate summary statistics for uploaded data"""
        pollutant_cols = [col for col in df.columns if col in self.pollutants]
        
        summary = {
            'total_rows': len(df),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d %H:%M'),
                'end': df.index.max().strftime('%Y-%m-%d %H:%M')
            },
            'pollutants': {},
            'missing_data': {},
            'data_quality': 'Good'
        }
        
        for pollutant in pollutant_cols:
            summary['pollutants'][pollutant] = {
                'mean': float(df[pollutant].mean()),
                'std': float(df[pollutant].std()),
                'min': float(df[pollutant].min()),
                'max': float(df[pollutant].max()),
                'median': float(df[pollutant].median())
            }
            
            missing_pct = (df[pollutant].isnull().sum() / len(df)) * 100
            summary['missing_data'][pollutant] = round(missing_pct, 2)
        
        # Assess data quality
        avg_missing = np.mean(list(summary['missing_data'].values()))
        if avg_missing > 20:
            summary['data_quality'] = 'Poor'
        elif avg_missing > 10:
            summary['data_quality'] = 'Fair'
        
        return summary
    
    def save_uploaded_file(self, df: pd.DataFrame, filename: str) -> Path:
        """Save uploaded and processed file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"uploaded_{timestamp}_{filename}"
        filepath = self.upload_dir / safe_filename
        
        df.to_csv(filepath)
        logger.info(f"Uploaded file saved to {filepath}")
        
        return filepath
    
    def check_model_compatibility(self, df: pd.DataFrame) -> Dict[str, bool]:
        """Check which models can be used with uploaded data"""
        available_pollutants = [p for p in self.pollutants if p in df.columns]
        
        compatibility = {
            pollutant: pollutant in available_pollutants
            for pollutant in self.pollutants
        }
        
        return compatibility