from abc import ABC, abstractmethod
import pandas as pd
import vectorbt as vbt
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class StrategyConfig:
    """Base configuration for all strategies"""
    rebalance_freq: str = 'M'  # Default monthly rebalancing
    initial_capital: float = 100.0
    size: float = 1.0
    size_type: str = 'percent'

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    start_date: str
    end_date: str
    rebalance_freq: str
    initial_capital: float
    size: float
    size_type: str

class StrategyBase(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, config: BacktestConfig):
        """Initialize strategy with configuration."""
        self.config = config
        if isinstance(config, dict):
            self.initial_capital = config['backtest_settings']['initial_capital']
            self.size = config['backtest_settings']['size']
            self.size_type = config['backtest_settings']['size_type']
        else:
            self.initial_capital = config.initial_capital
            self.size = config.size
            self.size_type = config.size_type
        self._portfolio = None
        self._signals = None
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals for the strategy.
        
        Args:
            data (pd.DataFrame): Price and other market data
            
        Returns:
            pd.Series: Boolean series with True for long positions
        """
        pass
    
    def backtest(self, price_series: pd.Series) -> vbt.Portfolio:
        """Run backtest for the strategy."""
        # Generate signals
        if isinstance(price_series, pd.Series):
            data = price_series.to_frame()
        else:
            data = price_series
        
        # Detect data frequency
        dates = data.index
        if len(dates) > 1:
            avg_days = (dates[-1] - dates[0]).days / (len(dates) - 1)
            freq = '1M' if avg_days > 25 else '1D'
            print(f"Using detected frequency: {freq}")
        else:
            freq = '1M'  # Default to monthly if can't determine
            print(f"Using default frequency: {freq}")
            
        # Ensure data is resampled to monthly if using monthly rebalancing
        if hasattr(self.config, 'rebalance_freq') and self.config.rebalance_freq == 'M':
            data = data.resample('M').last()
            price_series = price_series.resample('M').last()
            freq = '1M'  # Ensure frequency is monthly
            
        signals = self.generate_signals(data)
        
        # Ensure signals is a Series
        if isinstance(signals, pd.DataFrame):
            signals = signals.iloc[:, 0]
        
        # Create entries and exits
        entries = signals > 0  # Convert float signals to boolean for entries
        exits = ~(signals > 0)  # Convert float signals to boolean for exits
        
        # Ensure price series is valid
        price_series = price_series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(price_series) == 0:
            raise ValueError("No valid prices found in price series")
            
        # Normalize price series to start at 100
        price_series = 100 * price_series / price_series.iloc[0]
        
        # Create portfolio
        portfolio = vbt.Portfolio.from_signals(
            close=price_series,
            entries=entries,
            exits=exits,
            size=signals,  # Use original signals for position sizing
            init_cash=self.initial_capital,
            freq=freq,  # Use proper frequency string
            direction='longonly',  # Only long positions
            accumulate=False,  # Don't accumulate positions
            upon_long_conflict='ignore',  # Ignore new signals if already in position
            upon_dir_conflict='ignore',  # Ignore signals in opposite direction
            upon_opposite_entry='ignore',  # Ignore entry signals in opposite direction
        )
        
        self._portfolio = portfolio
        self._signals = signals
        
        return portfolio
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy performance statistics.
        
        Returns:
            Dict[str, Any]: Dictionary of performance metrics
        """
        if self._portfolio is None:
            raise ValueError("Must run backtest before getting stats")
        return self._portfolio.stats()
