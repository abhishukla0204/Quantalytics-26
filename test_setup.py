"""
Test Setup Script - Verify all dependencies are installed correctly
Run this before executing the main strategy
"""

import sys

print("="*60)
print("Quantalytics Project - Dependency Check")
print("="*60)
print()

# Track success
all_ok = True

# 1. Check Python version
print("1. Checking Python version...")
py_version = sys.version_info
print(f"   Python {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major >= 3 and py_version.minor >= 8:
    print("   ✓ Python version OK (>= 3.8)")
else:
    print("   ✗ Python version too old (need >= 3.8)")
    all_ok = False
print()

# 2. Check pandas
print("2. Checking pandas...")
try:
    import pandas as pd
    print(f"   pandas version: {pd.__version__}")
    print("   ✓ pandas installed")
except ImportError as e:
    print(f"   ✗ pandas not found: {e}")
    all_ok = False
print()

# 3. Check numpy
print("3. Checking numpy...")
try:
    import numpy as np
    print(f"   numpy version: {np.__version__}")
    print("   ✓ numpy installed")
except ImportError as e:
    print(f"   ✗ numpy not found: {e}")
    all_ok = False
print()

# 4. Check backtesting
print("4. Checking backtesting...")
try:
    import backtesting
    print(f"   backtesting version: {backtesting.__version__}")
    print("   ✓ backtesting installed")
except ImportError as e:
    print(f"   ✗ backtesting not found: {e}")
    print("   Install: pip install backtesting")
    all_ok = False
print()

# 5. Check talib (most problematic)
print("5. Checking TA-Lib...")
try:
    import talib
    print(f"   TA-Lib version: {talib.__version__}")
    print("   ✓ TA-Lib installed")
except ImportError as e:
    print(f"   ✗ TA-Lib not found: {e}")
    print()
    print("   INSTALLATION INSTRUCTIONS:")
    print("   " + "-"*50)
    print("   Option A (Recommended): Use conda")
    print("     conda install -c conda-forge ta-lib")
    print()
    print("   Option B: Download pre-built wheel from")
    print("     https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib")
    print("     pip install TA_Lib-0.4.XX-cpXX-cpXX-win_amd64.whl")
    print("   " + "-"*50)
    all_ok = False
print()

# 6. Check matplotlib
print("6. Checking matplotlib...")
try:
    import matplotlib
    print(f"   matplotlib version: {matplotlib.__version__}")
    print("   ✓ matplotlib installed")
except ImportError as e:
    print(f"   ✗ matplotlib not found: {e}")
    print("   Install: pip install matplotlib")
    all_ok = False
print()

# 7. Check data files
print("7. Checking data files...")
import os

data_files = [
    "XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv",
    "XAGUSD_M1/DAT_MT_XAGUSD_M1_2024.csv"
]

for file in data_files:
    if os.path.exists(file):
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"   ✓ {file} ({size_mb:.1f} MB)")
    else:
        print(f"   ✗ {file} not found")
        all_ok = False
print()

# 8. Quick functionality test
print("8. Testing core functionality...")
try:
    # Test data loading
    df_test = pd.DataFrame({
        'Open': [100, 101, 102],
        'High': [101, 102, 103],
        'Low': [99, 100, 101],
        'Close': [100.5, 101.5, 102.5],
        'Volume': [1000, 1100, 1200]
    }, index=pd.date_range('2024-01-01', periods=3, freq='1min'))
    
    # Test indicator calculation
    from backtesting.test import SMA
    sma = df_test['Close'].rolling(2).mean()
    
    print("   ✓ Data processing works")
    print("   ✓ Indicator calculation works")
except Exception as e:
    print(f"   ✗ Functionality test failed: {e}")
    all_ok = False
print()

# Final verdict
print("="*60)
if all_ok:
    print("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
    print()
    print("You're ready to run the strategy:")
    print("  python strategy.py XAUUSD_M1\\DAT_MT_XAUUSD_M1_2024.csv")
    print("  python strategy.py XAGUSD_M1\\DAT_MT_XAGUSD_M1_2024.csv")
else:
    print("✗✗✗ SOME CHECKS FAILED ✗✗✗")
    print()
    print("Please install missing dependencies before proceeding.")
    print("See SETUP_GUIDE.md for detailed instructions.")
print("="*60)

sys.exit(0 if all_ok else 1)
