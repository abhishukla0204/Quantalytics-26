# Quick Setup Guide for Quantalytics Project

## Windows Installation Steps

### 1. Install TA-Lib (Critical!)

TA-Lib is required but can be tricky to install on Windows.

**Option A: Using Conda (Easiest)**
```powershell
# Install Anaconda/Miniconda first from https://www.anaconda.com/
conda install -c conda-forge ta-lib
```

**Option B: Using Pre-built Wheel**
```powershell
# Download the appropriate wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# 
# For Python 3.8 64-bit: TA_Lib-0.4.24-cp38-cp38-win_amd64.whl
# For Python 3.9 64-bit: TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
# For Python 3.10 64-bit: TA_Lib-0.4.24-cp310-cp310-win_amd64.whl
# For Python 3.11 64-bit: TA_Lib-0.4.24-cp311-cp311-win_amd64.whl

# Then install:
pip install path\to\downloaded\TA_Lib-0.4.24-cp3XX-cp3XX-win_amd64.whl
```

### 2. Install Other Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Verify Installation
```powershell
python test_setup.py
```

### 4. Run the Strategy

**Test on Gold:**
```powershell
python strategy.py data\XAUUSD_M1\DAT_MT_XAUUSD_M1_2024.csv
```

**Test on Silver:**
```powershell
python strategy.py data\XAGUSD_M1\DAT_MT_XAGUSD_M1_2024.csv
```

---

## Troubleshooting

### Issue: "No module named 'talib'"
**Solution:** TA-Lib not installed. Follow Option A or B above.

### Issue: "ImportError: DLL load failed"
**Solution:** 
- Reinstall Visual C++ Redistributable: https://aka.ms/vs/17/release/vc_redist.x64.exe
- Or use conda installation method

### Issue: Strategy runs but no trades executed
**Solution:**
- Check if data loaded correctly (should show ~350k bars)
- Verify indicators are calculating (not all NaN)
- Try adjusting RSI thresholds (increase sensitivity)

### Issue: Memory error with large datasets
**Solution:**
- Close other applications
- Reduce data size by testing on subset: df = df[-100000:]

---


## Quick Commands Cheat Sheet

```powershell
# Install dependencies
pip install -r requirements.txt

# Test setup
python test_setup.py

# Run backtest on Gold
python strategy.py data\XAUUSD_M1\DAT_MT_XAUUSD_M1_2024.csv

# Run backtest on Silver
python strategy.py data\XAGUSD_M1\DAT_MT_XAGUSD_M1_2024.csv

# Check Python version
python --version

# List installed packages
pip list | findstr "pandas numpy backtesting talib"
```

---