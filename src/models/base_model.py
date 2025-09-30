from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Any
from src.utils.logger import logger

class BaseModel(ABC):
    """Abstract base class for all forecasting models"""
    
    def __init__(self, model_name: str, config: Dict[str, Any] = None):
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        self.training_history = {}
        
    @abstractmethod
    def build_model(self):
        """Build the model architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        mape = np.mean(np.abs((y_test - predictions) / (y_test + 1e-6))) * 100
        r2 = 1 - (np.sum((y_test - predictions) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2
        }
        
        logger.info(f"{self.model_name} Metrics: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save model to disk"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': self.config,
            'training_history': self.training_history,
            'model_name': self.model_name
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.config = model_data['config']
        self.training_history = model_data.get('training_history', {})
        self.model_name = model_data.get('model_name', self.model_name)
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance (if applicable)"""
        return None