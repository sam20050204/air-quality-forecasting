import pandas as pd
import numpy as np
from typing import List
from src.utils.logger import logger
from src.utils.config import config

class FeatureEngineer:
    """Create features for time series forecasting"""
    
    def __init__(self, data_config: dict = None):
        self.config = data_config or config.data_config
        self.feature_config = self.config.get('feature_engineering', {})
        
    def create_lag_features(self, df: pd.DataFrame, column: str, lags: List[int] = None) -> pd.DataFrame:
        """Create lag features for a given column"""
        lags = lags or self.feature_config.get('lag_features', [1, 3, 6, 12, 24])
        
        df_features = df.copy()
        for lag in lags:
            df_features[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame, column: str, windows: List[int] = None) -> pd.DataFrame:
        """Create rolling window statistics"""
        windows = windows or self.feature_config.get('rolling_windows', [3, 6, 12, 24])
        
        df_features = df.copy()
        
        for window in windows:
            # Rolling mean
            df_features[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
            
            # Rolling std
            df_features[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
            
            # Rolling min/max
            df_features[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
            df_features[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        return df_features
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime index"""
        df_features = df.copy()
        
        time_features = self.feature_config.get('time_features', ['hour', 'day_of_week', 'month'])
        
        if 'hour' in time_features:
            df_features['hour'] = df.index.hour
            # Cyclical encoding for hour
            df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
            df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
        
        if 'day_of_week' in time_features:
            df_features['day_of_week'] = df.index.dayofweek
            df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
            # Cyclical encoding
            df_features['dow_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
            df_features['dow_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
        
        if 'month' in time_features:
            df_features['month'] = df.index.month
            # Cyclical encoding
            df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        
        if 'season' in time_features:
            df_features['season'] = (df.index.month % 12 + 3) // 3
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame, pollutants: List[str]) -> pd.DataFrame:
        """Create interaction features between pollutants"""
        df_features = df.copy()
        
        # PM2.5/PM10 ratio
        if 'PM2.5' in pollutants and 'PM10' in pollutants:
            df_features['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1e-6)
        
        # Pollutant combinations
        if 'NO2' in pollutants and 'O3' in pollutants:
            df_features['NO2_O3_interaction'] = df['NO2'] * df['O3']
        
        return df_features
    
    def create_all_features(self, df: pd.DataFrame, target_column: str, 
                           include_lags: bool = True, 
                           include_rolling: bool = True,
                           include_time: bool = True,
                           include_interactions: bool = True) -> pd.DataFrame:
        """Create all features for the dataset"""
        logger.info("Creating features for model training")
        
        df_features = df.copy()
        
        # Time features
        if include_time:
            df_features = self.create_time_features(df_features)
        
        # Lag features
        if include_lags:
            df_features = self.create_lag_features(df_features, target_column)
        
        # Rolling features
        if include_rolling:
            df_features = self.create_rolling_features(df_features, target_column)
        
        # Interaction features
        if include_interactions:
            pollutants = config.data_config['pollutants']
            df_features = self.create_interaction_features(df_features, pollutants)
        
        # Drop NaN values created by lag and rolling features
        initial_len = len(df_features)
        df_features = df_features.dropna()
        logger.info(f"Dropped {initial_len - len(df_features)} rows with NaN values after feature engineering")
        
        return df_features
    
    def prepare_sequences(self, df: pd.DataFrame, target_column: str, 
                         lookback: int = 24, forecast_horizon: int = 1) -> tuple:
        """Prepare sequences for LSTM model"""
        logger.info(f"Preparing sequences with lookback={lookback}, forecast={forecast_horizon}")
        
        # Get feature columns (exclude target if it exists)
        feature_cols = [col for col in df.columns if col != target_column or col in df.columns]
        
        data = df[feature_cols].values
        target = df[target_column].values
        
        X, y = [], []
        
        for i in range(lookback, len(data) - forecast_horizon + 1):
            X.append(data[i - lookback:i])
            y.append(target[i + forecast_horizon - 1])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        
        return X, y
    
    def prepare_features_for_xgboost(self, df: pd.DataFrame, target_column: str, 
                                     lookback: int = 24) -> tuple:
        """Prepare features for XGBoost model"""
        logger.info(f"Preparing features for XGBoost with lookback={lookback}")
        
        df_features = df.copy()
        
        # Create lag features for target
        for lag in range(1, lookback + 1):
            df_features[f'{target_column}_lag_{lag}'] = df[target_column].shift(lag)
        
        # Create rolling features
        for window in [3, 6, 12, 24]:
            if window <= lookback:
                df_features[f'{target_column}_rolling_mean_{window}'] = df[target_column].rolling(window).mean()
                df_features[f'{target_column}_rolling_std_{window}'] = df[target_column].rolling(window).std()
        
        # Add time features if not already present
        if 'hour' not in df_features.columns:
            df_features = self.create_time_features(df_features)
        
        # Drop NaN rows
        df_features = df_features.dropna()
        
        # Separate features and target
        X = df_features.drop(columns=[target_column])
        y = df_features[target_column]
        
        logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        
        return X, y