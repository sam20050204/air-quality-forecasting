import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.data_preprocessor import DataPreprocessor
from src.data.feature_engineering import FeatureEngineer
from src.models.xgboost_model import XGBoostModel
from src.utils.logger import logger
from src.utils.config import config

def train_xgboost_model(data_path: str, target_pollutant: str = 'PM2.5', save_dir: str = 'models/saved_models'):
    """Train XGBoost model for air quality forecasting"""
    
    logger.info(f"Starting XGBoost training for {target_pollutant}")
    logger.info(f"Data path: {data_path}")
    
    # Step 1: Preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_pipeline(data_path, normalize=False)
    
    train_df = processed_data['train']
    val_df = processed_data['val']
    test_df = processed_data['test']
    
    # Step 2: Feature engineering
    feature_engineer = FeatureEngineer()
    
    # Prepare features for XGBoost
    lookback = config.model_config['xgboost'].get('lookback_hours', 24)
    
    logger.info("Creating features for training set...")
    X_train, y_train = feature_engineer.prepare_features_for_xgboost(
        train_df, target_pollutant, lookback=lookback
    )
    
    logger.info("Creating features for validation set...")
    X_val, y_val = feature_engineer.prepare_features_for_xgboost(
        val_df, target_pollutant, lookback=lookback
    )
    
    logger.info("Creating features for test set...")
    X_test, y_test = feature_engineer.prepare_features_for_xgboost(
        test_df, target_pollutant, lookback=lookback
    )
    
    # Step 3: Train model
    xgb_config = config.model_config['xgboost']
    model = XGBoostModel(xgb_config)
    
    logger.info("Training XGBoost model...")
    model.train(X_train, y_train, X_val, y_val)
    
    # Step 4: Evaluate model
    logger.info("Evaluating model on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    
    # Print metrics
    print("\n" + "="*50)
    print(f"XGBoost Model - {target_pollutant}")
    print("="*50)
    print(f"MAE:  {test_metrics['mae']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    print(f"R²:   {test_metrics['r2']:.3f}")
    print("="*50 + "\n")
    
    # Step 5: Save model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    model_filename = save_path / f'xgboost_{target_pollutant.lower()}.pkl'
    model.save_model(str(model_filename))
    
    # Save metrics
    metrics_filename = save_path / f'xgboost_{target_pollutant.lower()}_metrics.json'
    with open(metrics_filename, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    logger.info(f"Model and metrics saved to {save_path}")
    
    # Save feature names for later use
    feature_names_file = save_path / f'xgboost_{target_pollutant.lower()}_features.json'
    with open(feature_names_file, 'w') as f:
        json.dump(list(X_train.columns), f)
    
    return model, test_metrics

def train_all_pollutants(data_path: str, save_dir: str = 'models/saved_models'):
    """Train models for all pollutants"""
    
    pollutants = config.data_config['pollutants']
    results = {}
    
    for pollutant in pollutants:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training model for {pollutant}")
            logger.info('='*60)
            
            model, metrics = train_xgboost_model(data_path, pollutant, save_dir)
            results[pollutant] = {
                'model': model,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Error training model for {pollutant}: {str(e)}")
            results[pollutant] = {'error': str(e)}
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for pollutant, result in results.items():
        if 'error' in result:
            print(f"{pollutant}: FAILED - {result['error']}")
        else:
            metrics = result['metrics']
            print(f"{pollutant}: R²={metrics['r2']:.3f}, RMSE={metrics['rmse']:.2f}")
    print("="*60 + "\n")
    
    return results

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train air quality forecasting models')
    parser.add_argument('--data', type=str, default='data/raw/delhi_air_quality.csv',
                        help='Path to training data CSV file')
    parser.add_argument('--pollutant', type=str, default='all',
                        help='Pollutant to train (default: all)')
    parser.add_argument('--save_dir', type=str, default='models/saved_models',
                        help='Directory to save trained models')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not Path(args.data).exists():
        logger.error(f"Data file not found: {args.data}")
        print(f"Error: Data file '{args.data}' does not exist.")
        print("Please run 'python scripts/download_data.py' first to generate sample data.")
        sys.exit(1)
    
    # Train models
    if args.pollutant.lower() == 'all':
        logger.info("Training models for all pollutants...")
        train_all_pollutants(args.data, args.save_dir)
    else:
        logger.info(f"Training model for {args.pollutant}...")
        train_xgboost_model(args.data, args.pollutant, args.save_dir)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()