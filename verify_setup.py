import sys
from pathlib import Path

def verify_setup():
    checks = []
    
    # Check directories
    dirs = ['data/raw', 'models/saved_models', 'src/data', 'dashboard']
    for d in dirs:
        exists = Path(d).exists()
        checks.append((f"Directory {d}", exists))
    
    # Check files
    files = ['requirements.txt', 'configs/model_config.yaml']
    for f in files:
        exists = Path(f).exists()
        checks.append((f"File {f}", exists))
    
    # Check imports
    try:
        import pandas
        checks.append(("pandas installed", True))
    except:
        checks.append(("pandas installed", False))
    
    try:
        import streamlit
        checks.append(("streamlit installed", True))
    except:
        checks.append(("streamlit installed", False))
    
    # Print results
    print("\n=== Setup Verification ===\n")
    all_passed = True
    for check, status in checks:
        symbol = "✅" if status else "❌"
        print(f"{symbol} {check}")
        if not status:
            all_passed = False
    
    print("\n" + "="*30)
    if all_passed:
        print("✅ All checks passed! Setup complete.")
    else:
        print("❌ Some checks failed. Please review above.")
    
    return all_passed

if __name__ == "__main__":
    verify_setup()