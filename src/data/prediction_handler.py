import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import timedelta
from src.models.xgboost_model import XGBoostModel
from src.data.feature_engineering import FeatureEngineer
from src.utils.logger import logger
from src.utils.config import config
from src.utils.api_calculator import AQICalculator

class PredictionHandler:
    """Handle predictions on uploaded data"""
    
    def __init__(self, model_dir: str = 'models/saved_models'):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.feature_engineer = FeatureEngineer()
        self.pollutants = config.data_config['pollutants']
    
    def load_models(self, pollutants: Optional[List[str]] = None) -> Dict[str, bool]:
        """Load trained models for specified pollutants"""
        pollutants = pollutants or self.pollutants
        loaded = {}
        
        for pollutant in pollutants:
            model_path = self.model_dir / f'xgboost_{pollutant.lower()}.pkl'
            
            if model_path.exists():
                try:
                    model = XGBoostModel()
                    model.load_model(str(model_path))
                    self.models[pollutant] = model
                    loaded[pollutant] = True
                    logger.info(f"Loaded model for {pollutant}")
                except Exception as e:
                    logger.error(f"Failed to load model for {pollutant}: {e}")
                    loaded[pollutant] = False
            else:
                logger.warning(f"Model not found for {pollutant}")
                loaded[pollutant] = False
        
        return loaded
    
    def prepare_features(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            # Load feature names used during training
            feature_file = self.model_dir / f'xgboost_{pollutant.lower()}_features.json'
            
            if feature_file.exists():
                import json
                with open(feature_file, 'r') as f:
                    required_features = json.load(f)
            else:
                required_features = None
            
            # Create features
            lookback = config.model_config['xgboost'].get('lookback_hours', 24)
            X, y = self.feature_engineer.prepare_features_for_xgboost(
                df, pollutant, lookback=lookback
            )
            
            # Ensure features match training features
            if required_features:
                # Add missing features with zeros
                for feat in required_features:
                    if feat not in X.columns:
                        X[feat] = 0
                
                # Reorder columns to match training
                X = X[required_features]
            
            return X
            
        except Exception as e:
            logger.error(f"Error preparing features for {pollutant}: {e}")
            raise
    
    def predict_pollutant(self, df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
        """Make predictions for a single pollutant"""
        if pollutant not in self.models:
            raise ValueError(f"Model not loaded for {pollutant}")
        
        # Prepare features
        X = self.prepare_features(df, pollutant)
        
        # Make predictions
        predictions = self.models[pollutant].predict(X)
        
        # Create result dataframe
        result_df = pd.DataFrame({
            'datetime': X.index,
            f'{pollutant}_actual': df.loc[X.index, pollutant],
            f'{pollutant}_predicted': predictions
        })
        
        # Calculate error metrics
        result_df[f'{pollutant}_error'] = result_df[f'{pollutant}_predicted'] - result_df[f'{pollutant}_actual']
        result_df[f'{pollutant}_error_pct'] = (result_df[f'{pollutant}_error'] / result_df[f'{pollutant}_actual']) * 100
        
        return result_df
    
    def predict_all_pollutants(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Make predictions for all available pollutants"""
        results = {}
        available_pollutants = [p for p in self.pollutants if p in df.columns and p in self.models]
        
        for pollutant in available_pollutants:
            try:
                logger.info(f"Predicting {pollutant}")
                result_df = self.predict_pollutant(df, pollutant)
                results[pollutant] = result_df
            except Exception as e:
                logger.error(f"Error predicting {pollutant}: {e}")
                results[pollutant] = None
        
        return results
    
    def forecast_future(self, df: pd.DataFrame, pollutant: str, 
                       hours_ahead: int = 24) -> pd.DataFrame:
        """Forecast future values for a pollutant"""
        if pollutant not in self.models:
            raise ValueError(f"Model not loaded for {pollutant}")
        
        # Use last N hours as context
        lookback = config.model_config['xgboost'].get('lookback_hours', 24)
        context_df = df.tail(lookback * 2)  # Extra context for feature creation
        
        forecasts = []
        last_date = df.index[-1]
        
        # Iterative forecasting
        for i in range(hours_ahead):
            # Prepare features
            X = self.prepare_features(context_df, pollutant)
            
            if len(X) == 0:
                logger.warning(f"Insufficient data for forecasting at step {i}")
                break
            
            # Predict next value
            pred = self.models[pollutant].predict(X.tail(1))
            
            # Create forecast entry
            forecast_time = last_date + timedelta(hours=i+1)
            forecasts.append({
                'datetime': forecast_time,
                f'{pollutant}_forecast': pred[0]
            })
            
            # Update context with prediction
            new_row = context_df.iloc[-1:].copy()
            new_row.index = [forecast_time]
            new_row[pollutant] = pred[0]
            context_df = pd.concat([context_df, new_row])
        
        forecast_df = pd.DataFrame(forecasts)
        forecast_df.set_index('datetime', inplace=True)
        
        return forecast_df
    
    def calculate_metrics(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        """Calculate evaluation metrics for predictions"""
        metrics = {}
        
        for pollutant, result_df in results.items():
            if result_df is None:
                continue
            
            actual = result_df[f'{pollutant}_actual'].values
            predicted = result_df[f'{pollutant}_predicted'].values
            
            mae = np.mean(np.abs(predicted - actual))
            rmse = np.sqrt(np.mean((predicted - actual) ** 2))
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-6))) * 100
            r2 = 1 - (np.sum((actual - predicted) ** 2) / np.sum((actual - np.mean(actual)) ** 2))
            
            metrics[pollutant] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2),
                'samples': len(actual)
            }
        
        return metrics
    
    def calculate_aqi_from_predictions(self, results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate AQI from predicted pollutant values"""
        if 'PM2.5' not in results or results['PM2.5'] is None:
            return None
        
        pm25_df = results['PM2.5']
        aqi_data = []
        
        for idx, row in pm25_df.iterrows():
            predicted_pm25 = row['PM2.5_predicted']
            aqi = AQICalculator.calculate_aqi('PM2.5', predicted_pm25)
            category, color = AQICalculator.get_aqi_category(aqi)
            
            aqi_data.append({
                'datetime': idx,
                'AQI': aqi,
                'category': category,
                'PM2.5': predicted_pm25
            })
        
        aqi_df = pd.DataFrame(aqi_data)
        aqi_df.set_index('datetime', inplace=True)
        
        return aqi_df
    
    def generate_insights(self, df: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> List[str]:
        """Generate insights from predictions"""
        insights = []
        
        # Check for high pollution periods
        for pollutant, result_df in results.items():
            if result_df is None:
                continue
            
            predicted_col = f'{pollutant}_predicted'
            
            # Find high values
            threshold = result_df[predicted_col].quantile(0.9)
            high_periods = result_df[result_df[predicted_col] > threshold]
            
            if len(high_periods) > 0:
                insights.append(
                    f"⚠️ {pollutant} predicted to be high ({threshold:.1f}+ μg/m³) in "
                    f"{len(high_periods)} time periods"
                )
        
        # Check prediction accuracy
        metrics = self.calculate_metrics(results)
        for pollutant, metric in metrics.items():
            if metric['r2'] > 0.95:
                insights.append(f"✅ {pollutant} predictions are highly accurate (R² = {metric['r2']:.3f})")
            elif metric['r2'] < 0.80:
                insights.append(f"⚠️ {pollutant} predictions have lower accuracy (R² = {metric['r2']:.3f})")
        
        # Check trends
        for pollutant, result_df in results.items():
            if result_df is None:
                continue
            
            predicted_col = f'{pollutant}_predicted'
            
            # Calculate trend
            first_half = result_df[predicted_col].iloc[:len(result_df)//2].mean()
            second_half = result_df[predicted_col].iloc[len(result_df)//2:].mean()
            
            change_pct = ((second_half - first_half) / first_half) * 100
            
            if abs(change_pct) > 20:
                direction = "increasing" if change_pct > 0 else "decreasing"
                insights.append(f"📈 {pollutant} is {direction} by {abs(change_pct):.1f}% over the period")
        
        return insights