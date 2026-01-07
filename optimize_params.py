"""
Parameter Optimization Script
Automatically find optimal parameters for the trading strategy
"""

import pandas as pd
import numpy as np
from backtesting import Backtest
from strategy import AdaptiveMomentumReversion, load_data
import sys
import os
from itertools import product


def optimize_strategy(csv_path, max_iterations=50):
    """
    Optimize strategy parameters using grid search
    
    Args:
        csv_path: Path to CSV data file
        max_iterations: Maximum optimization iterations
    """
    print(f"\n{'='*70}")
    print(f"PARAMETER OPTIMIZATION - {os.path.basename(csv_path)}")
    print(f"{'='*70}\n")
    
    # Load data
    print("Loading data...")
    df = load_data(csv_path)
    print(f"Data loaded: {len(df)} bars from {df.index[0]} to {df.index[-1]}\n")
    
    # Split into train/test
    split_idx = int(len(df) * 0.75)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    print(f"Training set: {len(df_train)} bars ({df_train.index[0]} to {df_train.index[-1]})")
    print(f"Testing set: {len(df_test)} bars ({df_test.index[0]} to {df_test.index[-1]})\n")
    
    # Initialize backtest
    bt = Backtest(
        df_train,
        AdaptiveMomentumReversion,
        cash=100000,
        commission=0.00002,
        exclusive_orders=True
    )
    
    print("Starting optimization...")
    print("This may take several minutes...\n")
    
    # Define parameter ranges
    param_grid = {
        'rsi_period': [10, 14, 18],
        'rsi_os': [25, 30, 35],
        'rsi_ob': [65, 70, 75],
        'bb_period': [15, 20, 25],
        'bb_std': [1.5, 2.0, 2.5],
        'atr_period': [10, 14, 18],
        'sma_fast': [8, 10, 12],
        'sma_slow': [25, 30, 35],
        'sl_atr_mult': [1.5, 2.0, 2.5],
        'tp_atr_mult': [2.5, 3.0, 3.5]
    }
    
    # Run optimization
    try:
        stats = bt.optimize(
            rsi_period=param_grid['rsi_period'],
            rsi_os=param_grid['rsi_os'],
            rsi_ob=param_grid['rsi_ob'],
            bb_period=param_grid['bb_period'],
            bb_std=param_grid['bb_std'],
            atr_period=param_grid['atr_period'],
            sma_fast=param_grid['sma_fast'],
            sma_slow=param_grid['sma_slow'],
            sl_atr_mult=param_grid['sl_atr_mult'],
            tp_atr_mult=param_grid['tp_atr_mult'],
            maximize='Sharpe Ratio',
            constraint=lambda p: (
                p.rsi_period < p.bb_period and
                p.sma_fast < p.sma_slow and
                p.rsi_os < 50 and
                p.rsi_ob > 50 and
                p.sl_atr_mult < p.tp_atr_mult
            ),
            max_tries=max_iterations,
            return_heatmap=False
        )
    except Exception as e:
        print(f"Optimization failed: {e}")
        return None
    
    # Display optimal parameters
    print(f"\n{'='*70}")
    print("OPTIMAL PARAMETERS (Training Set)")
    print(f"{'='*70}")
    
    optimal_params = {
        'rsi_period': stats._strategy.rsi_period,
        'rsi_os': stats._strategy.rsi_os,
        'rsi_ob': stats._strategy.rsi_ob,
        'bb_period': stats._strategy.bb_period,
        'bb_std': stats._strategy.bb_std,
        'atr_period': stats._strategy.atr_period,
        'sma_fast': stats._strategy.sma_fast,
        'sma_slow': stats._strategy.sma_slow,
        'sl_atr_mult': stats._strategy.sl_atr_mult,
        'tp_atr_mult': stats._strategy.tp_atr_mult
    }
    
    for param, value in optimal_params.items():
        print(f"{param:20s}: {value}")
    
    print(f"\n{'='*70}")
    print("TRAINING SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"Sharpe Ratio          : {stats['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio         : {stats['Sortino Ratio']:.2f}")
    print(f"Total Return          : {stats['Return [%]']:.2f}%")
    print(f"Max Drawdown          : {stats['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate              : {stats['Win Rate [%]']:.2f}%")
    print(f"Total Trades          : {stats['# Trades']}")
    print(f"Avg Trade             : {stats['Avg. Trade [%]']:.2f}%")
    
    # Test on out-of-sample data
    print(f"\n{'='*70}")
    print("OUT-OF-SAMPLE VALIDATION")
    print(f"{'='*70}\n")
    print("Testing optimized parameters on unseen data...\n")
    
    # Create test backtest with optimal parameters
    class OptimizedStrategy(AdaptiveMomentumReversion):
        rsi_period = optimal_params['rsi_period']
        rsi_os = optimal_params['rsi_os']
        rsi_ob = optimal_params['rsi_ob']
        bb_period = optimal_params['bb_period']
        bb_std = optimal_params['bb_std']
        atr_period = optimal_params['atr_period']
        sma_fast = optimal_params['sma_fast']
        sma_slow = optimal_params['sma_slow']
        sl_atr_mult = optimal_params['sl_atr_mult']
        tp_atr_mult = optimal_params['tp_atr_mult']
    
    bt_test = Backtest(
        df_test,
        OptimizedStrategy,
        cash=100000,
        commission=0.00002,
        exclusive_orders=True
    )
    
    stats_test = bt_test.run()
    
    print(f"{'='*70}")
    print("TESTING SET PERFORMANCE")
    print(f"{'='*70}")
    print(f"Sharpe Ratio          : {stats_test['Sharpe Ratio']:.2f}")
    print(f"Sortino Ratio         : {stats_test['Sortino Ratio']:.2f}")
    print(f"Total Return          : {stats_test['Return [%]']:.2f}%")
    print(f"Max Drawdown          : {stats_test['Max. Drawdown [%]']:.2f}%")
    print(f"Win Rate              : {stats_test['Win Rate [%]']:.2f}%")
    print(f"Total Trades          : {stats_test['# Trades']}")
    print(f"Avg Trade             : {stats_test['Avg. Trade [%]']:.2f}%")
    
    # Calculate degradation
    sharpe_degradation = (stats['Sharpe Ratio'] - stats_test['Sharpe Ratio']) / stats['Sharpe Ratio'] * 100
    return_degradation = (stats['Return [%]'] - stats_test['Return [%]']) / stats['Return [%]'] * 100
    
    print(f"\n{'='*70}")
    print("OVERFITTING ANALYSIS")
    print(f"{'='*70}")
    print(f"Sharpe Ratio Degradation  : {sharpe_degradation:.1f}%")
    print(f"Return Degradation        : {return_degradation:.1f}%")
    
    if sharpe_degradation < 20:
        print("\n✓ LOW OVERFITTING - Strategy is robust!")
    elif sharpe_degradation < 40:
        print("\n⚠ MODERATE OVERFITTING - Use with caution")
    else:
        print("\n✗ HIGH OVERFITTING - Strategy may not generalize well")
    
    print(f"{'='*70}\n")
    
    # Save optimal parameters
    output_file = f"optimal_params_{os.path.basename(csv_path).replace('.csv', '.txt')}"
    with open(output_file, 'w') as f:
        f.write("# Optimal Strategy Parameters\n")
        f.write(f"# Optimized on: {df_train.index[0]} to {df_train.index[-1]}\n")
        f.write(f"# Validated on: {df_test.index[0]} to {df_test.index[-1]}\n\n")
        
        for param, value in optimal_params.items():
            f.write(f"{param} = {value}\n")
        
        f.write(f"\n# Training Performance\n")
        f.write(f"# Sharpe Ratio: {stats['Sharpe Ratio']:.2f}\n")
        f.write(f"# Total Return: {stats['Return [%]']:.2f}%\n")
        f.write(f"# Max Drawdown: {stats['Max. Drawdown [%]']:.2f}%\n")
        
        f.write(f"\n# Testing Performance\n")
        f.write(f"# Sharpe Ratio: {stats_test['Sharpe Ratio']:.2f}\n")
        f.write(f"# Total Return: {stats_test['Return [%]']:.2f}%\n")
        f.write(f"# Max Drawdown: {stats_test['Max. Drawdown [%]']:.2f}%\n")
    
    print(f"Optimal parameters saved to: {output_file}\n")
    
    return stats, stats_test, optimal_params


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Usage: python optimize_params.py <path_to_csv> [max_iterations]")
        print("\nExamples:")
        print("  python optimize_params.py XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv")
        print("  python optimize_params.py XAGUSD_M1/DAT_MT_XAGUSD_M1_2024.csv 100")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Run optimization
    result = optimize_strategy(csv_path, max_iterations=max_iter)
    
    if result is None:
        print("Optimization failed!")
        sys.exit(1)
    
    print("Optimization completed successfully!")
    print("\nTo use these parameters, update the values in strategy.py")


if __name__ == "__main__":
    main()
