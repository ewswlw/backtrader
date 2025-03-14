import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class MAConfig:
    """Configuration for MA strategy."""
    ma_window: int = 20
    entry_threshold: float = 0.0

class MovingAverageStrategy(StrategyBase):
    """Moving average crossover strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MA strategy."""
        # Create a full config for the parent class using the settings from the actual config file
        if 'backtest_settings' in config:
            # Original format with full config structure
            super().__init__(BacktestConfig(**config['backtest_settings']))
            if 'strategies' in config and 'MA' in config['strategies']:
                self.ma_config = MAConfig(**config['strategies']['MA'])
            else:
                self.ma_config = MAConfig()
        else:
            # Simplified format with just the MA parameters directly
            # We need to pass the parent config separately
            import os
            
            # Create a simple BacktestConfig with default values
            backtest_config = BacktestConfig(
                start_date="",
                end_date="",
                rebalance_freq="M",
                initial_capital=100.0,
                size=1.0,
                size_type="percent"
            )
            super().__init__(backtest_config)
            
            # Use the passed config directly for MA parameters
            self.ma_config = MAConfig(
                ma_window=config.get('ma_window', 20),
                entry_threshold=config.get('entry_threshold', 0.0)
            )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on moving average crossover."""
        # Get price data
        prices = data.iloc[:, 0].astype(float)
        
        # Calculate moving average
        ma = prices.rolling(window=self.ma_config.ma_window, min_periods=1).mean()
        
        # Generate signals
        signals = pd.Series(False, index=prices.index)
        
        # Long when price above MA, short when below
        signals[prices > (ma + self.ma_config.entry_threshold)] = True
        signals[prices < (ma - self.ma_config.entry_threshold)] = False
        
        # Fill any missing values with previous signal
        signals = signals.astype(bool)
        return signals

    def optimize_parameters(self, price_series: pd.Series) -> Dict[str, Any]:
        """Optimize MA window parameter."""
        results = []
        
        print("\nOptimizing MA window...")
        # Test MA windows from 1 to 60 months
        for window in range(1, 61):
            self.ma_config.ma_window = window
            signals = self.generate_signals(pd.DataFrame(price_series))
            
            # Calculate strategy returns
            returns = price_series.pct_change().fillna(0)
            strategy_returns = signals.shift(1) * returns  # Shift signals to avoid lookahead bias
            
            # Calculate metrics
            total_return = (1 + strategy_returns).prod() - 1
            ann_vol = strategy_returns.std() * np.sqrt(12)  # Annualize monthly volatility
            sharpe = (total_return / ann_vol) if ann_vol != 0 else 0
            
            results.append({
                'window': window,
                'total_return': total_return,
                'ann_vol': ann_vol,
                'sharpe': sharpe
            })
            
            if window % 10 == 0:
                print(f"Tested window size: {window}")
        
        # Convert results to DataFrame for analysis
        results_df = pd.DataFrame(results)
        
        # Find optimal window based on Sharpe ratio
        optimal_window = results_df.loc[results_df['sharpe'].idxmax(), 'window']
        print(f"\nOptimal MA window: {optimal_window}")
        print(f"Max Sharpe ratio: {results_df['sharpe'].max():.2f}")
        
        # Update strategy parameter
        self.ma_config.ma_window = int(optimal_window)
        
        return {
            'ma_window': int(optimal_window),
            'sharpe': results_df['sharpe'].max(),
            'total_return': results_df.loc[results_df['sharpe'].idxmax(), 'total_return']
        }
