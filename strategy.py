"""
Quantalytics Trading Strategy - Prometeo 2026
Gold (XAU/USD) and Silver (XAG/USD) Trading System
using namespace std; // Pseudocode style guide
"""

import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import talib as ta
import sys
import os


class AdaptiveMomentumReversion(Strategy):
    # Strategy parameters - optimized for 5-min resampled data
    rsi_period = 14          # Standard RSI
    rsi_ob = 68              # Slightly sensitive overbought
    rsi_os = 32              # Slightly sensitive oversold
    bb_period = 20           # Standard BB
    bb_std = 1.8             # Tighter bands
    atr_period = 14          # Standard ATR
    sma_fast = 12            # Fast MA
    sma_slow = 26            # Slow MA
    vol_period = 20          # Volatility window
    
    # Risk management
    risk_pct = 0.015         # Risk 1.5% per trade
    sl_atr_mult = 1.5        # Stop loss: 1.5x ATR
    tp_atr_mult = 2.5        # Take profit: 2.5x ATR (1:1.67 R:R)
    max_trades_per_day = 10  # Max trades per day
    
    def init(self):
        # Price data
        p = self.data.Close
        h = self.data.High
        l = self.data.Low
        c = self.data.Close
        o = self.data.Open
        
        # Technical indicators
        self.rsi = self.I(self._rsi, p, self.rsi_period)
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(
            self._bollinger_bands, p, self.bb_period, self.bb_std
        )
        self.atr = self.I(self._atr, h, l, c, self.atr_period)
        self.sma_f = self.I(SMA, p, self.sma_fast)
        self.sma_s = self.I(SMA, p, self.sma_slow)
        self.vol = self.I(self._volatility, p, self.vol_period)
        self.macd_line, self.macd_signal, _ = self.I(
            self._macd, p, 12, 26, 9
        )
        
        # Trade tracking
        self.daily_trades = 0
        self.last_trade_date = None
        self.entry_price = None
        self.sl_price = None
        self.tp_price = None
    
    def _rsi(self, p, n):
        """Relative Strength Index"""
        delta = pd.Series(p).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=n).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=n).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).values
    
    def _bollinger_bands(self, p, n, std):
        """Bollinger Bands"""
        s = pd.Series(p)
        mid = s.rolling(window=n).mean()
        sd = s.rolling(window=n).std()
        upper = mid + (std * sd)
        lower = mid - (std * sd)
        return upper.values, mid.values, lower.values
    
    def _atr(self, h, l, c, n):
        """Average True Range"""
        high = pd.Series(h)
        low = pd.Series(l)
        close = pd.Series(c)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=n).mean().values
    
    def _volatility(self, p, n):
        """Rolling volatility"""
        return pd.Series(p).pct_change().rolling(window=n).std().values
    
    def _macd(self, p, fast, slow, signal):
        """MACD indicator"""
        s = pd.Series(p)
        ema_fast = s.ewm(span=fast, adjust=False).mean()
        ema_slow = s.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return macd.values, sig.values, hist.values
    
    def _calculate_position_size(self, entry_p, sl_p):
        """Dynamic position sizing based on risk percentage"""
        if sl_p is None or sl_p == 0:
            return 0.95
        
        risk_amount = self.equity * self.risk_pct
        risk_per_unit = abs(entry_p - sl_p)
        
        if risk_per_unit == 0:
            return 0.95
        
        units = risk_amount / risk_per_unit
        max_units = self.equity / entry_p
        size = min(units / max_units, 0.95)
        
        return max(0.1, min(size, 0.95))
    
    def _reset_daily_counter(self, current_date):
        """Reset daily trade counter"""
        if self.last_trade_date is None or current_date != self.last_trade_date:
            self.daily_trades = 0
            self.last_trade_date = current_date
    
    def _check_trade_limit(self):
        """Check if we can place more trades today"""
        return self.daily_trades < self.max_trades_per_day
    
    def _generate_signals(self):
        """
        Clean signal generation - Trend Following with Mean Reversion entries
        """
        p = self.data.Close[-1]
        
        # Trend identification
        sma_bull = self.sma_f[-1] > self.sma_s[-1]
        sma_bear = self.sma_f[-1] < self.sma_s[-1]
        
        # Mean reversion entry points (pullbacks in trend)
        rsi_oversold = self.rsi[-1] < self.rsi_os
        rsi_overbought = self.rsi[-1] > self.rsi_ob
        bb_lower_touch = p <= self.bb_lower[-1]
        bb_upper_touch = p >= self.bb_upper[-1]
        
        # MACD for momentum confirmation
        macd_bull = self.macd_line[-1] > self.macd_signal[-1]
        macd_bear = self.macd_line[-1] < self.macd_signal[-1]
        
        # LONG: Buy pullbacks in uptrend
        # Uptrend (SMA) + oversold/BB lower (pullback) + MACD turning up
        long_signal = (
            sma_bull and                         # In uptrend
            (rsi_oversold or bb_lower_touch) and # At pullback level
            macd_bull                            # Momentum turning up
        )
        
        # SHORT: Sell rallies in downtrend
        # Downtrend (SMA) + overbought/BB upper (rally) + MACD turning down
        short_signal = (
            sma_bear and                         # In downtrend
            (rsi_overbought or bb_upper_touch) and # At rally level
            macd_bear                            # Momentum turning down
        )
        
        return long_signal, short_signal
    
    def next(self):
        """Main trading logic executed on each bar"""
        # Get current date for trade limiting
        current_date = self.data.index[-1].date()
        self._reset_daily_counter(current_date)
        
        # Check trade limit
        if not self._check_trade_limit():
            return
        
        p = self.data.Close[-1]
        atr = self.atr[-1]
        
        # Exit logic: Stop loss & Take profit
        if self.position:
            if self.position.is_long:
                if p <= self.sl_price or p >= self.tp_price:
                    self.position.close()
                    self.daily_trades += 1
                    return
            elif self.position.is_short:
                if p >= self.sl_price or p <= self.tp_price:
                    self.position.close()
                    self.daily_trades += 1
                    return
        
        # Entry logic
        if not self.position:
            long_sig, short_sig = self._generate_signals()
            
            if long_sig and not short_sig:
                # Calculate stop loss and take profit for long
                self.entry_price = p
                self.sl_price = p - (self.sl_atr_mult * atr)
                self.tp_price = p + (self.tp_atr_mult * atr)
                
                # Calculate position size
                size = self._calculate_position_size(self.entry_price, self.sl_price)
                
                # Enter long position
                self.buy(size=size)
                self.daily_trades += 1
            
            elif short_sig and not long_sig:
                # Calculate stop loss and take profit for short
                self.entry_price = p
                self.sl_price = p + (self.sl_atr_mult * atr)
                self.tp_price = p - (self.tp_atr_mult * atr)
                
                # Calculate position size
                size = self._calculate_position_size(self.entry_price, self.sl_price)
                
                # Enter short position
                self.sell(size=size)
                self.daily_trades += 1


def load_data(csv_path, resample='5min'):
    """Load and prepare OHLC data from CSV, with optional resampling"""
    df = pd.read_csv(
        csv_path,
        names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )
    # Combine Date and Time columns
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('Datetime')
    df = df.drop(['Date', 'Time'], axis=1)
    df = df.sort_index()
    
    # Resample to reduce noise (M1 -> 5min by default)
    if resample:
        df = df.resample(resample).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
    
    return df


def calculate_metrics(stats):
    """Extract and display key performance metrics"""
    metrics = {
        'Sharpe Ratio': stats['Sharpe Ratio'],
        'Sortino Ratio': stats['Sortino Ratio'],
        'Max Drawdown': stats['Max. Drawdown [%]'],
        'Win Rate': stats['Win Rate [%]'],
        'Total Return': stats['Return [%]'],
        'Total Trades': stats['# Trades'],
        'Avg Trade': stats['Avg. Trade [%]'],
        'Max Trade Duration': stats['Max. Trade Duration'],
        'Avg Trade Duration': stats['Avg. Trade Duration'],
        'Profit Factor': stats.get('Profit Factor', 'N/A'),
        'Exposure Time': stats['Exposure Time [%]'],
    }
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")
    print("="*60 + "\n")
    
    return metrics


def run_backtest(csv_path, cash=100000, commission_pct=0.00002):
    """
    Main backtest execution
    Commission: $2 or 0.002% (whichever is lower)
    For typical trade sizes, 0.002% = 0.00002 in decimal
    """
    print(f"\nLoading data from: {csv_path}")
    df = load_data(csv_path)
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    
    # Initialize backtest
    bt = Backtest(
        df,
        AdaptiveMomentumReversion,
        cash=cash,
        commission=commission_pct,
        exclusive_orders=True,
        trade_on_close=False
    )
    
    print("\nRunning backtest...")
    stats = bt.run()
    
    # Calculate and display metrics
    metrics = calculate_metrics(stats)
    
    # Generate plot
    print("\nGenerating performance plot...")
    bt.plot()
    
    return stats, bt


def main():
    """Main execution function"""
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python strategy.py <path_to_csv>")
        print("\nExample:")
        print("  python strategy.py XAUUSD_M1/DAT_MT_XAUUSD_M1_2024.csv")
        print("  python strategy.py XAGUSD_M1/DAT_MT_XAGUSD_M1_2024.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    # Run backtest
    stats, bt = run_backtest(csv_path, cash=100000, commission_pct=0.00002)
    
    # Save results
    output_file = f"backtest_results_{os.path.basename(csv_path).replace('.csv', '.html')}"
    print(f"\nSaving results to: {output_file}")
    
    return stats


if __name__ == "__main__":
    main()
