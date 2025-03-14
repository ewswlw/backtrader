"""
Multi-Regime Adaptive Strategy (MRAS)
Combines multiple factors with regime detection for adaptive trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from dataclasses import dataclass
from .strategy_base import StrategyBase, BacktestConfig

@dataclass
class MRASConfig:
    """Configuration for MRAS strategy."""
    # Regime detection parameters
    regime_ma_window: int = 60  # Window for regime detection
    regime_vol_window: int = 20  # Window for volatility calculation
    
    # Signal generation parameters
    momentum_window: int = 20  # Window for momentum calculation
    hy_ma_window: int = 5  # Window for HY signal
    yield_ma_window: int = 20  # Window for yield curve signal
    
    # Thresholds
    vol_threshold: float = 0.15  # Annualized volatility threshold
    momentum_threshold: float = 0.0  # Momentum threshold
    
    # Position sizing
    vol_target: float = 0.10  # Target annualized volatility
    max_leverage: float = 1.0  # Maximum leverage

class MRASStrategy(StrategyBase):
    """Multi-Regime Adaptive Strategy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MRAS strategy."""
        # Create a full config for the parent class
        if 'backtest_settings' in config:
            super().__init__(BacktestConfig(**config['backtest_settings']))
            if 'strategies' in config and 'MRAS' in config['strategies']:
                self.mras_config = MRASConfig(**config['strategies']['MRAS'])
            else:
                self.mras_config = MRASConfig()
        else:
            # Use default config
            backtest_config = BacktestConfig(
                start_date="",
                end_date="",
                rebalance_freq="M",
                initial_capital=100.0,
                size=1.0,
                size_type="percent"
            )
            super().__init__(backtest_config)
            self.mras_config = MRASConfig()
    
    def detect_regime(self, data: pd.DataFrame) -> pd.Series:
        """Detect market regime using multiple indicators."""
        # 1. Volatility regime
        returns = data['cad_ig_er_index'].pct_change()
        vol = returns.rolling(window=self.mras_config.regime_vol_window).std() * np.sqrt(252)
        vol_regime = vol < self.mras_config.vol_threshold
        
        # 2. Credit spread regime
        spread_ma = data['cad_oas'].rolling(window=self.mras_config.regime_ma_window).mean()
        spread_regime = data['cad_oas'] < spread_ma
        
        # 3. Yield curve regime
        curve_ma = data['us_3m_10y'].rolling(window=self.mras_config.yield_ma_window).mean()
        curve_regime = data['us_3m_10y'] > curve_ma
        
        # 4. HY signal regime
        hy_ma = data['us_hy_er_index'].rolling(window=self.mras_config.hy_ma_window).mean()
        hy_regime = data['us_hy_er_index'] > hy_ma
        
        # Combine regimes (at least 3 out of 4 must be positive)
        regime_count = vol_regime.astype(int) + spread_regime.astype(int) + \
                      curve_regime.astype(int) + hy_regime.astype(int)
        
        return regime_count >= 3

    def calculate_position_sizes(self, data: pd.DataFrame, base_signals: pd.Series) -> pd.Series:
        """Calculate position sizes using volatility targeting."""
        returns = data['cad_ig_er_index'].pct_change()
        vol = returns.rolling(window=self.mras_config.regime_vol_window).std() * np.sqrt(252)
        
        # Calculate scaling factor based on target volatility
        vol_scalar = np.minimum(
            self.mras_config.vol_target / vol.replace(0, np.inf),
            self.mras_config.max_leverage
        )
        
        return base_signals * vol_scalar

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals with position sizing."""
        # Calculate momentum
        momentum = data['cad_ig_er_index'].pct_change(self.mras_config.momentum_window)
        momentum_signal = momentum > self.mras_config.momentum_threshold
        
        # Get regime signal
        regime_signal = self.detect_regime(data)
        
        # Combine signals
        base_signals = (momentum_signal & regime_signal).astype(float)
        
        # Apply position sizing
        return self.calculate_position_sizes(data, base_signals)

    def optimize_parameters(self, price_series: pd.Series) -> Dict[str, Any]:
        """Optimize strategy parameters."""
        results = []
        
        print("\nOptimizing MRAS parameters...")
        
        # Test different regime MA windows
        for regime_window in [40, 60, 80]:
            for momentum_window in [10, 20, 30]:
                for hy_window in [5, 10, 15]:
                    self.mras_config.regime_ma_window = regime_window
                    self.mras_config.momentum_window = momentum_window
                    self.mras_config.hy_ma_window = hy_window
                    
                    signals = self.generate_signals(pd.DataFrame(price_series))
                    
                    # Calculate strategy returns
                    returns = price_series.pct_change().fillna(0)
                    strategy_returns = signals.shift(1) * returns
                    
                    # Calculate metrics
                    total_return = (1 + strategy_returns).prod() - 1
                    ann_vol = strategy_returns.std() * np.sqrt(12)
                    sharpe = (total_return / ann_vol) if ann_vol != 0 else 0
                    
                    results.append({
                        'regime_window': regime_window,
                        'momentum_window': momentum_window,
                        'hy_window': hy_window,
                        'total_return': total_return,
                        'sharpe': sharpe,
                        'volatility': ann_vol
                    })
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        # Find best parameters by Sharpe ratio
        best_params = results_df.loc[results_df['sharpe'].idxmax()]
        
        return {
            'regime_ma_window': int(best_params['regime_window']),
            'momentum_window': int(best_params['momentum_window']),
            'hy_ma_window': int(best_params['hy_window']),
            'total_return': best_params['total_return'],
            'sharpe': best_params['sharpe']
        }
