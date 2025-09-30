import pandas as pd
import requests
from pathlib import Path

def download_sample_data():
    """Download sample air quality dataset"""
    
    # Create data directory
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # You can use this sample dataset from Kaggle or government sources
    # For now, let's create a sample dataset
    
    print("Creating sample dataset...")
    
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='H')
    
    data = pd.DataFrame({
        'datetime': dates,
        'city': 'Delhi',
        'PM2.5': np.random.uniform(10, 200, len(dates)),
        'PM10': np.random.uniform(20, 300, len(dates)),
        'NO2': np.random.uniform(5, 100, len(dates)),
        'SO2': np.random.uniform(2, 50, len(dates)),
        'CO': np.random.uniform(0.1, 5, len(dates)),
        'O3': np.random.uniform(10, 150, len(dates)),
    })
    
    # Save to CSV
    data.to_csv('data/raw/delhi_air_quality.csv', index=False)
    print("✅ Sample data created: data/raw/delhi_air_quality.csv")
    
    return data

if __name__ == "__main__":
    import numpy as np
    download_sample_data()


### 6.2 Run the Script
