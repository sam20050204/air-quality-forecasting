#!/usr/bin/env python3
"""
Diagnostic script to check model and data status
"""
from pathlib import Path

print("=" * 60)
print("AIR QUALITY FORECASTING - DIAGNOSTIC CHECK")
print("=" * 60)

# Check data directory
print("\n1. Checking Data Files:")
data_dir = Path('data/raw')
if data_dir.exists():
    files = list(data_dir.glob('*.csv'))
    if files:
        for f in files:
            size = f.stat().st_size / 1024  # KB
            print(f"   ✅ {f.name} ({size:.1f} KB)")
    else:
        print("   ❌ No CSV files found in data/raw/")
        print("   → Run: python scripts/download_data.py")
else:
    print("   ❌ data/raw/ directory not found")

# Check models directory
print("\n2. Checking Trained Models:")
models_dir = Path('models/saved_models')
if models_dir.exists():
    pkl_files = list(models_dir.glob('*.pkl'))
    json_files = list(models_dir.glob('*.json'))
    
    if pkl_files:
        print(f"   ✅ Found {len(pkl_files)} model files:")
        for f in pkl_files:
            size = f.stat().st_size / 1024  # KB
            print(f"      - {f.name} ({size:.1f} KB)")
    else:
        print("   ❌ No .pkl model files found")
        print("   → Run: python scripts/train_models.py --pollutant all")
    
    if json_files:
        print(f"   ✅ Found {len(json_files)} metrics files")
else:
    print("   ❌ models/saved_models/ directory not found")

# Check config files
print("\n3. Checking Configuration Files:")
config_files = ['configs/model_config.yaml', 'configs/data_config.yaml', 'configs/app_config.yaml']
for cf in config_files:
    if Path(cf).exists():
        print(f"   ✅ {cf}")
    else:
        print(f"   ❌ {cf} - MISSING")

# Check source files
print("\n4. Checking Source Code:")
source_files = [
    'src/models/xgboost_model.py',
    'src/models/base_model.py',
    'src/data/data_preprocessor.py',
    'src/utils/config.py'
]
for sf in source_files:
    if Path(sf).exists():
        print(f"   ✅ {sf}")
    else:
        print(f"   ❌ {sf} - MISSING")

# Try to import modules
print("\n5. Testing Module Imports:")
try:
    from src.utils.config import config
    print("   ✅ Config module imported successfully")
    print(f"      - Pollutants: {config.pollutants}")
except Exception as e:
    print(f"   ❌ Config import failed: {e}")

try:
    from src.models.xgboost_model import XGBoostModel
    print("   ✅ XGBoostModel imported successfully")
except Exception as e:
    print(f"   ❌ XGBoostModel import failed: {e}")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

# Check if ready to run
data_ready = Path('data/raw/delhi_air_quality.csv').exists()
models_ready = len(list(Path('models/saved_models').glob('*.pkl'))) > 0 if Path('models/saved_models').exists() else False

if data_ready and models_ready:
    print("✅ System is ready! Run: streamlit run dashboard/app.py")
elif data_ready and not models_ready:
    print("⚠️  Data found but models not trained")
    print("→ Run: python scripts/train_models.py --pollutant all")
elif not data_ready:
    print("⚠️  No data found")
    print("→ Run: python scripts/download_data.py")
else:
    print("❌ System not ready. Check errors above.")

print("=" * 60)