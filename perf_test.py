"""Quick performance test for strategy improvement"""
import pandas as pd
from backtesting import Backtest
import sys
sys.path.insert(0, '.')
from strategy import AdaptiveMomentumReversion, load_data

# Load data
df = load_data('data/XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv', resample='15min')

# Run backtest
bt = Backtest(df, AdaptiveMomentumReversion, cash=100000, 
              commission=0.00002, exclusive_orders=True, finalize_trades=True)
stats = bt.run()

print('CURRENT STRATEGY PERFORMANCE:')
print(f"Return: {stats['Return [%]']:.2f}%")
print(f"Sharpe: {stats['Sharpe Ratio']:.3f}")
print(f"Sortino: {stats['Sortino Ratio']:.3f}")
print(f"Max DD: {stats['Max. Drawdown [%]']:.2f}%")
print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
print(f"Trades: {stats['# Trades']}")
print(f"Avg Trade: {stats['Avg. Trade [%]']:.3f}%")
print(f"Profit Factor: {stats['Profit Factor']:.2f}")
