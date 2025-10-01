#!/usr/bin/env python3
"""
Quick fix script to diagnose and resolve dashboard issues
Run this before starting the dashboard
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def check_and_fix_data():
    """Check and generate sample data if missing"""
    print("="*60)
    print("1. Checking Data Files...")
    print("="*60)
    
    data_file = Path('data/raw/delhi_air_quality.csv')
    
    if not data_file.exists():
        print("❌ Sample data not found. Generating...")
        
        # Create directory
        data_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate 2 years of hourly data
        dates = pd.date_range(
            start='2023-01-01',
            end='2024-12-31',
            freq='H'
        )
        
        # Create realistic air quality data with patterns
        hours = dates.hour
        days = dates.dayofyear
        
        df = pd.DataFrame({
            'datetime': dates,
            'city': 'Delhi',
            'PM2.5': 50 + 30*np.sin(days/365*2*np.pi) + 20*np.sin(hours/24*2*np.pi) + np.random.normal(0, 10, len(dates)),
            'PM10': 80 + 50*np.sin(days/365*2*np.pi) + 30*np.sin(hours/24*2*np.pi) + np.random.normal(0, 15, len(dates)),
            'NO2': 30 + 15*np.sin(days/365*2*np.pi) + 10*np.sin(hours/24*2*np.pi) + np.random.normal(0, 5, len(dates)),
            'SO2': 15 + 8*np.sin(days/365*2*np.pi) + 5*np.sin(hours/24*2*np.pi) + np.random.normal(0, 3, len(dates)),
            'CO': 1.5 + 0.8*np.sin(days/365*2*np.pi) + 0.5*np.sin(hours/24*2*np.pi) + np.random.normal(0, 0.2, len(dates)),
            'O3': 45 + 20*np.sin(days/365*2*np.pi) + 15*np.sin(hours/24*2*np.pi) + np.random.normal(0, 5, len(dates)),
        })
        
        # Ensure positive values
        for col in ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']:
            df[col] = df[col].clip(lower=0.1)
        
        df.to_csv(data_file, index=False)
        print(f"✅ Generated sample data: {len(df)} records")
        print(f"   Saved to: {data_file}")
    else:
        df = pd.read_csv(data_file)
        print(f"✅ Data file found: {len(df)} records")
    
    return True

def check_models():
    """Check if models exist"""
    print("\n" + "="*60)
    print("2. Checking Models...")
    print("="*60)
    
    models_dir = Path('models/saved_models')
    
    if not models_dir.exists():
        print("❌ Models directory not found")
        print("   Run: python scripts/train_models.py --pollutant all")
        return False
    
    pkl_files = list(models_dir.glob('*.pkl'))
    
    if not pkl_files:
        print("❌ No trained models found")
        print("   Run: python scripts/train_models.py --pollutant all")
        return False
    
    print(f"✅ Found {len(pkl_files)} trained models:")
    for f in pkl_files:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")
    
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\n" + "="*60)
    print("3. Checking Dependencies...")
    print("="*60)
    
    required = {
        'streamlit': 'streamlit',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'plotly': 'plotly',
        'xgboost': 'xgboost',
        'scikit-learn': 'sklearn',
        'pyyaml': 'yaml'
    }
    
    missing = []
    
    for display_name, import_name in required.items():
        try:
            __import__(import_name)
            print(f"✅ {display_name}")
        except ImportError:
            print(f"❌ {display_name} - NOT INSTALLED")
            missing.append(display_name)
    
    if missing:
        print(f"\n⚠️  Install missing packages:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def check_config_files():
    """Check configuration files"""
    print("\n" + "="*60)
    print("4. Checking Configuration...")
    print("="*60)
    
    config_files = [
        'configs/model_config.yaml',
        'configs/data_config.yaml',
        'configs/app_config.yaml'
    ]
    
    all_exist = True
    for cf in config_files:
        if Path(cf).exists():
            print(f"✅ {cf}")
        else:
            print(f"❌ {cf} - MISSING")
            all_exist = False
    
    return all_exist

def provide_next_steps(has_data, has_models, has_deps, has_config):
    """Provide next steps based on checks"""
    print("\n" + "="*60)
    print("SUMMARY & NEXT STEPS")
    print("="*60)
    
    if not has_deps:
        print("\n❌ DEPENDENCIES MISSING")
        print("→ Run: pip install -r requirements.txt")
        return
    
    if not has_config:
        print("\n❌ CONFIGURATION FILES MISSING")
        print("→ Ensure all files from the project are uploaded")
        return
    
    if not has_data:
        print("\n⚠️  DATA NOT FOUND")
        print("→ Run this script again, it will generate sample data")
        return
    
    if not has_models:
        print("\n⚠️  MODELS NOT TRAINED")
        print("→ Train models: python scripts/train_models.py --pollutant all")
        print("   (This will take 5-10 minutes)")
        return
    
    print("\n✅ SYSTEM READY!")
    print("→ Start dashboard: streamlit run dashboard/app.py")
    print("\nDashboard will be available at: http://localhost:8501")

def main():
    """Main diagnostic function"""
    print("\n" + "="*60)
    print("AIR QUALITY FORECASTING SYSTEM - QUICK FIX")
    print("="*60 + "\n")
    
    # Run checks
    has_data = check_and_fix_data()
    has_models = check_models()
    has_deps = check_dependencies()
    has_config = check_config_files()
    
    # Provide guidance
    provide_next_steps(has_data, has_models, has_deps, has_config)
    
    print("\n" + "="*60)
    print("For detailed diagnostics, run: python check_models.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()