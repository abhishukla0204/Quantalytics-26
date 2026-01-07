"""
Quick Test Runner - Test strategy on a subset of data for rapid validation
Useful for debugging and quick parameter testing
"""

import pandas as pd
import numpy as np
from backtesting import Backtest
from strategy import AdaptiveMomentumReversion, load_data, calculate_metrics
import sys
import os


def quick_test(csv_path, n_bars=50000, cash=100000):
    """
    Run backtest on a subset of data for quick validation
    
    Args:
        csv_path: Path to CSV file
        n_bars: Number of bars to test (default: 50,000)
        cash: Starting capital
    """
    print(f"\n{'='*70}")
    print(f"QUICK TEST - {os.path.basename(csv_path)}")
    print(f"{'='*70}\n")
    
    # Load full data
    print("Loading data...")
    df_full = load_data(csv_path)
    print(f"Full dataset: {len(df_full)} bars")
    
    # Take last n_bars for testing (most recent data)
    df = df_full.iloc[-n_bars:]
    print(f"Testing on last {len(df)} bars")
    print(f"Period: {df.index[0]} to {df.index[-1]}\n")
    
    # Run backtest
    print("Running backtest...")
    bt = Backtest(
        df,
        AdaptiveMomentumReversion,
        cash=cash,
        commission=0.00002,
        exclusive_orders=True,
        trade_on_close=False
    )
    
    stats = bt.run()
    
    # Display metrics
    metrics = calculate_metrics(stats)
    
    # Additional analysis
    print("\nTRADE ANALYSIS")
    print("="*70)
    
    if stats['# Trades'] > 0:
        trades_per_day = stats['# Trades'] / ((df.index[-1] - df.index[0]).days + 1)
        trades_per_month = trades_per_day * 30
        
        print(f"Average Trades per Day    : {trades_per_day:.1f}")
        print(f"Estimated Trades per Month: {trades_per_month:.1f}")
        
        # Check compliance
        print("\nCOMPLIANCE CHECK")
        print("="*70)
        
        if trades_per_month >= 10:
            print("✓ Minimum frequency met (>= 10 trades/month)")
        else:
            print("✗ Below minimum frequency (< 10 trades/month)")
            print("  Consider adjusting RSI thresholds or other parameters")
        
        if trades_per_day < 100:
            print("✓ Daily limit respected (< 100 trades/day)")
        else:
            print("✗ Exceeds daily limit (>= 100 trades/day)")
            print("  Daily limiter should prevent this - check code")
        
        # Performance assessment
        print("\nPERFORMANCE ASSESSMENT")
        print("="*70)
        
        sharpe = stats['Sharpe Ratio']
        if sharpe >= 1.5:
            print(f"✓ Excellent Sharpe Ratio: {sharpe:.2f} (>= 1.5)")
        elif sharpe >= 1.0:
            print(f"⚠ Good Sharpe Ratio: {sharpe:.2f} (>= 1.0)")
        else:
            print(f"✗ Poor Sharpe Ratio: {sharpe:.2f} (< 1.0)")
            print("  Consider parameter optimization")
        
        max_dd = stats['Max. Drawdown [%]']
        if max_dd >= -25:
            print(f"✓ Acceptable Drawdown: {max_dd:.1f}% (>= -25%)")
        else:
            print(f"✗ Excessive Drawdown: {max_dd:.1f}% (< -25%)")
            print("  Review risk management settings")
        
        win_rate = stats['Win Rate [%]']
        if win_rate >= 45:
            print(f"✓ Good Win Rate: {win_rate:.1f}% (>= 45%)")
        else:
            print(f"⚠ Low Win Rate: {win_rate:.1f}% (< 45%)")
            print("  May still be profitable if profit factor is high")
    else:
        print("⚠ No trades executed!")
        print("\nPossible reasons:")
        print("  1. Data quality issues (all NaN indicators)")
        print("  2. Parameters too restrictive")
        print("  3. Insufficient data for indicator calculation")
        print("\nSuggestions:")
        print("  - Check data loading")
        print("  - Reduce RSI thresholds (e.g., 35/65)")
        print("  - Increase n_bars parameter")
    
    print("\n" + "="*70)
    
    # Option to plot
    try:
        user_input = input("\nGenerate interactive plot? (y/n): ").lower()
        if user_input == 'y':
            print("Generating plot...")
            bt.plot()
    except:
        print("Skipping plot generation")
    
    return stats


def compare_assets():
    """Compare strategy performance on Gold vs Silver"""
    print(f"\n{'='*70}")
    print("COMPARATIVE ANALYSIS: GOLD vs SILVER")
    print(f"{'='*70}\n")
    
    assets = {
        'Gold': 'XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv',
        'Silver': 'XAGUSD_M1/DAT_MT_XAGUSD_M1_2024.csv'
    }
    
    results = {}
    
    for name, path in assets.items():
        if not os.path.exists(path):
            print(f"⚠ {name} data not found: {path}")
            continue
        
        print(f"\nTesting {name}...")
        print("-" * 70)
        
        try:
            stats = quick_test(path, n_bars=50000, cash=100000)
            results[name] = stats
        except Exception as e:
            print(f"✗ {name} test failed: {e}")
            continue
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}\n")
        
        metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Return [%]', 
                   'Max. Drawdown [%]', 'Win Rate [%]', '# Trades']
        
        print(f"{'Metric':<25} {'Gold':<15} {'Silver':<15} {'Winner':<10}")
        print("-" * 70)
        
        for metric in metrics:
            gold_val = results['Gold'].get(metric, 0)
            silver_val = results['Silver'].get(metric, 0)
            
            # Determine winner (lower is better for drawdown)
            if 'Drawdown' in metric:
                winner = 'Gold' if gold_val > silver_val else 'Silver'
            else:
                winner = 'Gold' if gold_val > silver_val else 'Silver'
            
            print(f"{metric:<25} {gold_val:>10.2f}    {silver_val:>10.2f}    {winner:<10}")
        
        print("="*70)


def main():
    """Main execution"""
    if len(sys.argv) < 2:
        print("Quick Test Runner - Test strategy on data subset\n")
        print("Usage:")
        print("  python quick_test.py <path_to_csv> [n_bars]")
        print("  python quick_test.py compare\n")
        print("Examples:")
        print("  python quick_test.py XAUUSD_M1\\DAT_MT_XAUUSD_M1_2024.csv")
        print("  python quick_test.py XAGUSD_M1\\DAT_MT_XAGUSD_M1_2024.csv 100000")
        print("  python quick_test.py compare  # Compare Gold vs Silver\n")
        sys.exit(1)
    
    if sys.argv[1].lower() == 'compare':
        compare_assets()
    else:
        csv_path = sys.argv[1]
        n_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 50000
        
        if not os.path.exists(csv_path):
            print(f"Error: File not found: {csv_path}")
            sys.exit(1)
        
        quick_test(csv_path, n_bars=n_bars)


if __name__ == "__main__":
    main()
