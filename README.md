# Quantalytics Trading Strategy - Prometeo 2026

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Backtesting.py](https://img.shields.io/badge/Backtesting.py-0.3.3+-green.svg)](https://kernc.github.io/backtesting.py/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Adaptive Momentum-Reversion Hybrid Strategy** for trading Gold (XAU/USD) and Silver (XAG/USD) markets.

---

## 🎯 Overview

This project implements a quantitative trading strategy for the **Quantalytics competition** at Prometeo 2026. The strategy combines:

- **Mean Reversion** (RSI + Bollinger Bands)
- **Momentum Trading** (MACD + SMA Crossover)
- **Dynamic Risk Management** (ATR-based stops, position sizing)
- **Transaction Cost Optimization**

**Target Assets:**
- Gold (XAU/USD) - 1-minute bars
- Silver (XAG/USD) - 1-minute bars

---

## ✨ Features

### Core Strategy Components

**Hybrid Signal Generation**
- RSI (14-period) for overbought/oversold detection
- Bollinger Bands (20-period, 2 std) for statistical extremes
- MACD (12, 26, 9) for momentum confirmation
- SMA Crossover (10/30) for trend alignment

**Risk Management**
- Dynamic position sizing (2% risk per trade)
- ATR-based stop-loss (2x ATR)
- ATR-based take-profit (3x ATR)
- Daily trade limit (max 100 trades/day)

**Cost Optimization**
- Commission: $2 or 0.002% (whichever is lower)
- Slippage consideration
- Minimal trading frequency to maximize profitability

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- TA-Lib (requires separate installation)

### Step 1: Clone the Repository


### Step 2: Install TA-Lib (Windows)

**Option A: Using conda (Recommended)**
```bash
conda install -c conda-forge ta-lib
```

**Option B: Using pre-built wheel**
```bash
# Download from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib-0.4.24-cp38-cp38-win_amd64.whl
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import backtesting; import talib; print('All packages installed successfully!')"
```

---

## 📊 Usage

### Basic Usage

Run the strategy on Gold data:

```bash
python strategy.py data/XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv
```

Run the strategy on Silver data:

```bash
python strategy.py data/XAGUSD_M1/DAT_MT_XAGUSD_M1_2024.csv
```

## 🧠 Strategy Logic

### Entry Conditions

#### Long Position (BUY)
```python
# Mean Reversion Signal
(RSI < 30 OR Price <= Lower Bollinger Band)

AND

# Momentum Confirmation
MACD Line > MACD Signal

AND

# Volatility Filter
Current Volatility > 50% of 50-period average

AND

# Trend Alignment
Fast SMA (10) > Slow SMA (30)
```

#### Short Position (SELL)
```python
# Mean Reversion Signal
(RSI > 70 OR Price >= Upper Bollinger Band)

AND

# Momentum Confirmation
MACD Line < MACD Signal

AND

# Volatility Filter
Current Volatility > 50% of 50-period average

AND

# Trend Alignment
Fast SMA (10) < Slow SMA (30)
```

### Exit Conditions

**Stop Loss:** Entry Price ± (2 × ATR)  
**Take Profit:** Entry Price ± (3 × ATR)  
**Risk-Reward Ratio:** 1:1.5

### Position Sizing

```python
Risk Amount = Account Equity × 2%
Risk Per Unit = |Entry Price - Stop Loss Price|
Position Size = Risk Amount / Risk Per Unit
Maximum Position = 95% of equity
```

---

## 📈 Performance Metrics

The strategy outputs the following key metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Sharpe Ratio** | Risk-adjusted return | > 1.5 |
| **Sortino Ratio** | Downside risk-adjusted return | > 1.8 |
| **Max Drawdown** | Largest peak-to-trough decline | < 25% |
| **Win Rate** | Percentage of winning trades | 45-55% |
| **Profit Factor** | Gross profit / Gross loss | > 1.5 |
| **Total Trades** | Number of completed trades | 10-50/month |
| **Avg Trade** | Average return per trade | Positive |
| **Exposure Time** | % of time in market | 40-60% |

---

## 📁 Project Structure

```
Prometeo/
│
├── strategy.py                      # Main trading strategy script
├── requirements.txt                 # Python dependencies
├── RESEARCH_REPORT.md               # Detailed methodology documentation
├── README.md                        # This file
│
└── data/                            # Market data folder
    ├── XAUUSD_M1/                   # Gold market data
    │   ├── DAT_MT_XAUUSD_M1_2024.csv
    │   └── DAT_MT_XAUUSD_M1_2024.txt
    │
    └── XAGUSD_M1/                   # Silver market data
        ├── DAT_MT_XAGUSD_M1_2024.csv
        └── DAT_MT_XAGUSD_M1_2024.txt
```

### Key Files

- **`strategy.py`**: Complete backtesting implementation with `backtesting.py`
- **`RESEARCH_REPORT.md`**: Comprehensive documentation of methodology, assumptions, and signal logic
- **`requirements.txt`**: All necessary Python packages with versions

---

## 🔧 Advanced Usage

### Optimizing Parameters

Modify strategy parameters in `strategy.py`:

```python
class AdaptiveMomentumReversion(Strategy):
    # Optimize these parameters
    rsi_period = 14      # RSI lookback
    rsi_ob = 70          # RSI overbought threshold
    rsi_os = 30          # RSI oversold threshold
    bb_period = 20       # Bollinger Bands period
    bb_std = 2.0         # Bollinger Bands standard deviation
    atr_period = 14      # ATR period
    sma_fast = 10        # Fast SMA period
    sma_slow = 30        # Slow SMA period
    risk_pct = 0.02      # Risk percentage per trade
    sl_atr_mult = 2.0    # Stop loss multiplier
    tp_atr_mult = 3.0    # Take profit multiplier
```

### Parameter Optimization Example

```python
from backtesting import Backtest

# Load data
df = load_data('data/XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv')

# Initialize backtest
bt = Backtest(df, AdaptiveMomentumReversion, cash=100000, commission=0.00002)

# Optimize parameters
stats = bt.optimize(
    rsi_period=range(10, 20, 2),
    bb_period=range(15, 30, 5),
    atr_period=range(10, 20, 2),
    maximize='Sharpe Ratio',
    constraint=lambda p: p.rsi_period < p.bb_period
)

print(stats)
```

### Walk-Forward Analysis

For robust out-of-sample testing:

```python
# Split data
train_data = df[:'2024-09-30']  # 75% for training
test_data = df['2024-10-01':]   # 25% for testing

# Train on in-sample data
bt_train = Backtest(train_data, AdaptiveMomentumReversion, cash=100000)
stats_train = bt_train.run()

# Validate on out-of-sample data
bt_test = Backtest(test_data, AdaptiveMomentumReversion, cash=100000)
stats_test = bt_test.run()

# Compare performance
print(f"In-Sample Sharpe: {stats_train['Sharpe Ratio']:.2f}")
print(f"Out-of-Sample Sharpe: {stats_test['Sharpe Ratio']:.2f}")
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. TA-Lib Installation Error

**Error:** `Cannot find TA-Lib`

**Solution:**
- Windows: Install from pre-built wheel (see Installation section)
- macOS: `brew install ta-lib && pip install TA-Lib`
- Linux: `sudo apt-get install ta-lib && pip install TA-Lib`

#### 2. Memory Error with Large Datasets

**Error:** `MemoryError` when loading CSV

**Solution:**
```python
# Load data in chunks
df = pd.read_csv('data/XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv', 
                 chunksize=50000)
```

#### 3. Plot Not Displaying

**Error:** Plot window not showing

**Solution:**
- Ensure you have a GUI backend: `pip install PyQt5`
- Or use: `bt.plot(open_browser=True)` to open in browser

#### 4. Commission Too High

**Error:** Strategy loses money due to high commissions

**Solution:**
- Verify commission is set correctly: `commission=0.00002` (0.002%)
- Reduce trade frequency by adjusting RSI thresholds

---

## 📚 Additional Resources

### Documentation

- [Strategy Research Report](RESEARCH_REPORT.md) - Detailed methodology
- [Backtesting.py Docs](https://kernc.github.io/backtesting.py/) - Framework documentation
- [TA-Lib Docs](https://mrjbq7.github.io/ta-lib/) - Technical analysis library

### Recommended Reading

1. **Quantitative Trading** by Ernest Chan
2. **Algorithmic Trading** by Andreas Clenow
3. **Evidence-Based Technical Analysis** by David Aronson
4. **Market Microstructure Theory** by Maureen O'Hara

---

### Enhancement Ideas

- [ ] Machine learning signal filtering
- [ ] Multi-timeframe analysis (M1, M5, M15)
- [ ] Sentiment analysis integration
- [ ] Portfolio optimization (Gold-Silver correlation trading)
- [ ] Monte Carlo simulation for risk assessment

---
