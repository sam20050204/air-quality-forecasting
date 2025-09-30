#!/usr/bin/env python3
"""
Setup script for Air Quality Forecasting System
Creates necessary directories and files
"""

import os
from pathlib import Path

def create_directory_structure():
    """Create all necessary directories"""
    
    directories = [
        # Data directories
        'data/raw',
        'data/processed',
        'data/uploaded',
        
        # Model directories
        'models/saved_models',
        
        # Source code directories
        'src/data',
        'src/models',
        'src/utils',
        'src/api',
        'src/evaluation',
        
        # Dashboard directories
        'dashboard/components',
        'dashboard/pages',
        
        # Scripts directory
        'scripts',
        
        # Config directory
        'configs',
        
        # Tests directory
        'tests',
        
        # Logs directory
        'logs'
    ]
    
    print("Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
        
        # Create .gitkeep for empty directories
        if directory.startswith('data/') or directory == 'logs':
            gitkeep = Path(directory) / '.gitkeep'
            gitkeep.touch(exist_ok=True)

def create_init_files():
    """Create __init__.py files for Python packages"""
    
    init_locations = [
        'src/__init__.py',
        'src/data/__init__.py',
        'src/models/__init__.py',
        'src/utils/__init__.py',
        'src/api/__init__.py',
        'src/evaluation/__init__.py',
        'dashboard/__init__.py',
        'dashboard/components/__init__.py',
        'dashboard/pages/__init__.py',
        'tests/__init__.py'
    ]
    
    print("\nCreating __init__.py files...")
    for init_file in init_locations:
        Path(init_file).touch(exist_ok=True)
        print(f"✓ Created: {init_file}")

def create_readme_files():
    """Create README files for key directories"""
    
    readmes = {
        'data/raw/README.md': """# Raw Data
        
Place raw air quality datasets here.

Supported formats:
- CSV files with datetime and pollutant columns
- Excel files (.xlsx, .xls)

Expected columns:
- datetime: timestamp of measurement
- city: city name
- PM2.5, PM10, NO2, SO2, CO, O3: pollutant concentrations
""",
        'data/processed/README.md': """# Processed Data

Preprocessed datasets are stored here automatically.

Files generated:
- Normalized datasets
- Train/validation/test splits
- Engineered features
""",
        'models/saved_models/README.md': """# Saved Models

Trained model files are stored here.

File naming convention:
- `{model_type}_{pollutant}.pkl` (e.g., xgboost_pm2.5.pkl)
- `{model_type}_{pollutant}_metrics.json` (evaluation metrics)
"""
    }
    
    print("\nCreating README files...")
    for filepath, content in readmes.items():
        with open(filepath, 'w') as f:
            f.write(content.strip())
        print(f"✓ Created: {filepath}")

def check_requirements():
    """Check if required packages are installed"""
    
    required_packages = [
        'pandas',
        'numpy',
        'scikit-learn',
        'xgboost',
        'streamlit',
        'plotly',
        'pyyaml'
    ]
    
    print("\nChecking required packages...")
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install them using: pip install -r requirements.txt")
    else:
        print("\n✅ All required packages are installed!")

def display_next_steps():
    """Display next steps for the user"""
    
    next_steps = """
╔════════════════════════════════════════════════════════════════╗
║                     SETUP COMPLETED! 🎉                        ║
╚════════════════════════════════════════════════════════════════╝

Next Steps:

1. Install Dependencies (if not done already):
   pip install -r requirements.txt

2. Generate Sample Data:
   python scripts/download_data.py

3. Train Models:
   python scripts/train_models.py --pollutant all

4. Launch Dashboard:
   streamlit run dashboard/app.py

5. Verify Setup:
   python verify_setup.py

Additional Commands:

- Train specific pollutant:
  python scripts/train_models.py --pollutant PM2.5

- View logs:
  tail -f logs/app.log

- Run tests:
  pytest tests/

Documentation:
- See README.md for detailed information
- Check configs/ directory for configuration options

Happy forecasting! 🌍
"""
    
    print(next_steps)

def main():
    """Main setup function"""
    
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     Air Quality Forecasting System - Project Setup            ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Create __init__.py files
    create_init_files()
    
    # Create README files
    create_readme_files()
    
    # Check requirements
    check_requirements()
    
    # Display next steps
    display_next_steps()

if __name__ == "__main__":
    main()