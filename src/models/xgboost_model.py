import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any
from src.models.base_model import BaseModel
from src.utils.logger import logger

class XGBoostModel(BaseModel):
    """XGBoost model for air quality forecasting"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__('XGBoost', config)
        self.build_model()
    
    def build_model(self):
        """Build XGBoost model with configuration"""
        params = {
            'n_estimators': self.config.get('n_estimators', 100),
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.1),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.model = xgb.XGBRegressor(**params)
        logger.info(f"XGBoost model built with params: {params}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the XGBoost model"""
        logger.info(f"Training XGBoost model on {len(X_train)} samples")
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, 
            y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Store training history
        self.training_history = {
            'n_samples': len(X_train),
            'n_features': X_train.shape[1],
            'feature_names': list(X_train.columns) if isinstance(X_train, pd.DataFrame) else None
        }
        
        logger.info("XGBoost training completed")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained and self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_next_steps(self, X_initial, n_steps: int = 24):
        """Predict multiple steps ahead using recursive prediction"""
        predictions = []
        X_current = X_initial.copy()
        
        for _ in range(n_steps):
            # Predict next value
            pred = self.predict(X_current[-1:])
            predictions.append(pred[0])
            
            # Update features for next prediction
            # This is a simplified version - in practice, you'd update lag features
            # based on the prediction
        
        return np.array(predictions)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if not self.is_trained:
            return None
        
        importance = self.model.feature_importances_
        feature_names = self.training_history.get('feature_names', 
                                                   [f'feature_{i}' for i in range(len(importance))])
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df