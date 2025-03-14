import pandas as pd
import numpy as np
from typing import Dict, Any
import vectorbt as vbt

class HYTimingStrategy:
    """Strategy that times CAD IG based on US HY excess returns MA signal."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        self.config = config
        
        # Handle both full config and simplified config structure
        if 'strategies' in config and 'HYTiming' in config['strategies']:
            # Full config structure
            self.ma_window = config['strategies']['HYTiming'].get('ma_window', 5)
        elif 'window' in config:
            # Simplified config structure
            self.ma_window = config['window']
            
            # Use a local copy of config
            import os
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'config', 'backtest_config.yaml')
            
            # We'll use default parameter values if config can't be loaded
            try:
                import yaml
                with open(config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
            except Exception as e:
                print(f"Warning: Could not load config file. Using default values: {str(e)}")
                self.config = {
                    'backtest_settings': {
                        'initial_capital': 100.0
                    },
                    'data': {
                        'file_path': 'backtest_data.csv'
                    }
                }
        else:
            # Default
            self.ma_window = 5
            
        self._data = None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on US HY excess returns MA."""
        # Calculate MA for US HY excess returns
        us_hy_ma = self._data['us_hy_er_index'].rolling(window=self.ma_window).mean()
        
        # Generate signals (1 when above MA, 0 when below)
        signals = (self._data['us_hy_er_index'] > us_hy_ma).astype(float)
        
        return signals

    def backtest(self, price_series: pd.Series) -> pd.Series:
        """Run backtest and return portfolio values."""
        # Load the full dataset to get US HY data
        data_path = self.config['data']['file_path']
        self._data = pd.read_csv(data_path)
        self._data['Date'] = pd.to_datetime(self._data['Date'])
        self._data.set_index('Date', inplace=True)
        
        # Optimize parameters first
        optimal_params = self.optimize_parameters(price_series)
        self.ma_window = optimal_params['ma_window']
        print(f"\nOptimal MA window: {self.ma_window}")
        print(f"Optimized Return: {optimal_params['total_return']:.2%}")
        
        # Generate signals using optimal parameters
        signals = self.generate_signals(self._data)
        
        # Align signals with price series
        signals = signals.reindex(price_series.index)
        
        # Calculate portfolio value
        returns = price_series.pct_change().fillna(0)
        strategy_returns = signals.shift(1) * returns  # Shift signals to avoid lookahead bias
        portfolio_value = pd.Series(index=price_series.index, dtype=float)
        portfolio_value.iloc[0] = self.config['backtest_settings']['initial_capital']
        
        for i in range(1, len(portfolio_value)):
            portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + strategy_returns.iloc[i])
        
        return portfolio_value

    def optimize_parameters(self, price_series: pd.Series) -> Dict[str, Any]:
        """Optimize MA window parameter."""
        results = []
        
        print("\nOptimizing MA window...")
        # Test MA windows from 1 to 60
        for window in range(1, 61):
            self.ma_window = window
            signals = self.generate_signals(self._data)
            
            # Align signals with price series
            signals = signals.reindex(price_series.index)
            
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
                'sharpe': sharpe,
                'volatility': ann_vol
            })
            
            print(f"Window {window:2d}: Return = {total_return:7.2%}, Sharpe = {sharpe:5.2f}")
        
        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results)
        
        # Sort by total return and get top 5
        print("\nTop 5 windows by Total Return:")
        top_by_return = results_df.nlargest(5, 'total_return')
        print(top_by_return.to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 100 else f"{x:.2f}"))
        
        # Sort by Sharpe and get top 5
        print("\nTop 5 windows by Sharpe Ratio:")
        top_by_sharpe = results_df.nlargest(5, 'sharpe')
        print(top_by_sharpe.to_string(float_format=lambda x: f"{x:.2%}" if abs(x) < 100 else f"{x:.2f}"))
        
        # Get the best window based on total return
        best_result = results_df.loc[results_df['total_return'].idxmax()]
        best_window = int(best_result['window'])
        best_return = best_result['total_return']
        
        # Set the best window and return the optimal parameters
        self.ma_window = best_window
        return {'ma_window': best_window, 'total_return': best_return}
