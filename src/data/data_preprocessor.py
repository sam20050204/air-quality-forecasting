import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.utils.logger import logger
from src.utils.config import config

class DataPreprocessor:
    """Preprocess air quality data for model training"""
    
    def __init__(self, data_config: dict = None):
        self.config = data_config or config.data_config
        self.pollutants = self.config['pollutants']
        self.time_settings = self.config['time_settings']
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw air quality data"""
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Convert datetime column
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values")
        
        # Check missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        for col, pct in missing_pct.items():
            if pct > 0:
                logger.info(f"{col}: {pct:.2f}% missing")
        
        # Interpolate missing values for pollutants
        method = self.time_settings.get('fill_method', 'interpolate')
        
        if method == 'interpolate':
            df[self.pollutants] = df[self.pollutants].interpolate(
                method='time', limit_direction='both'
            )
        elif method == 'forward_fill':
            df[self.pollutants] = df[self.pollutants].fillna(method='ffill')
        
        # Drop any remaining NaN values
        df.dropna(subset=self.pollutants, inplace=True)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Remove outliers from the dataset"""
        logger.info("Removing outliers")
        
        if method == 'iqr':
            for pollutant in self.pollutants:
                Q1 = df[pollutant].quantile(0.25)
                Q3 = df[pollutant].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                # Cap outliers instead of removing
                df[pollutant] = df[pollutant].clip(lower_bound, upper_bound)
        
        return df
    
    def resample_data(self, df: pd.DataFrame, freq: Optional[str] = None) -> pd.DataFrame:
        """Resample data to specified frequency"""
        freq = freq or self.time_settings['frequency']
        
        if freq != 'H':  # If not already hourly
            logger.info(f"Resampling data to {freq} frequency")
            df = df.resample(freq).mean()
        
        return df
    
    def normalize_data(self, df: pd.DataFrame, method: str = 'minmax') -> Tuple[pd.DataFrame, dict]:
        """Normalize pollutant values"""
        logger.info(f"Normalizing data using {method} method")
        
        scaler_params = {}
        df_normalized = df.copy()
        
        for pollutant in self.pollutants:
            if method == 'minmax':
                min_val = df[pollutant].min()
                max_val = df[pollutant].max()
                df_normalized[pollutant] = (df[pollutant] - min_val) / (max_val - min_val)
                scaler_params[pollutant] = {'min': min_val, 'max': max_val, 'method': 'minmax'}
            
            elif method == 'standard':
                mean_val = df[pollutant].mean()
                std_val = df[pollutant].std()
                df_normalized[pollutant] = (df[pollutant] - mean_val) / std_val
                scaler_params[pollutant] = {'mean': mean_val, 'std': std_val, 'method': 'standard'}
        
        return df_normalized, scaler_params
    
    def inverse_normalize(self, values: np.ndarray, pollutant: str, scaler_params: dict) -> np.ndarray:
        """Inverse transform normalized values"""
        params = scaler_params[pollutant]
        
        if params['method'] == 'minmax':
            return values * (params['max'] - params['min']) + params['min']
        elif params['method'] == 'standard':
            return values * params['std'] + params['mean']
        
        return values
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        split_config = self.config['train_test_split']
        test_size = split_config['test_size']
        val_size = split_config['validation_size']
        
        n = len(df)
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def preprocess_pipeline(self, filepath: str, normalize: bool = True) -> dict:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline")
        
        # Load data
        df = self.load_data(filepath)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove outliers
        df = self.remove_outliers(df)
        
        # Resample
        df = self.resample_data(df)
        
        # Split data before normalization
        train_df, val_df, test_df = self.split_data(df)
        
        # Normalize if requested
        scaler_params = None
        if normalize:
            train_df, scaler_params = self.normalize_data(train_df)
            
            # Apply same normalization to val and test
            for pollutant in self.pollutants:
                params = scaler_params[pollutant]
                if params['method'] == 'minmax':
                    val_df[pollutant] = (val_df[pollutant] - params['min']) / (params['max'] - params['min'])
                    test_df[pollutant] = (test_df[pollutant] - params['min']) / (params['max'] - params['min'])
                elif params['method'] == 'standard':
                    val_df[pollutant] = (val_df[pollutant] - params['mean']) / params['std']
                    test_df[pollutant] = (test_df[pollutant] - params['mean']) / params['std']
        
        logger.info("Preprocessing pipeline completed")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'scaler_params': scaler_params,
            'original': df
        }