from typing import Dict, Any
import pandas as pd
import vectorbt as vbt

from .strategy_base import StrategyBase, BacktestConfig

class BuyAndHoldStrategy(StrategyBase):
    """Simple buy and hold strategy that buys at the start and holds until the end."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the strategy with configuration."""
        super().__init__(BacktestConfig(**config['backtest_settings']))
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate buy and hold signals - buy at start, hold until end."""
        signals = pd.Series(True, index=data.index)
        signals.iloc[0] = False  # Don't buy on the first day
        signals.iloc[1:] = True  # Buy and hold from second day onwards
        return signals
