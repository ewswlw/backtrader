"""
Integrated script that combines raw_stats.py and strategy_comparison.py into one script.
This script runs strategies, calculates statistics, and generates comparisons and visualizations.
"""

import os
import pandas as pd
import numpy as np
import sys
import vectorbt as vbt
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# Add project directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import strategies
from src.strategies.ma_strategy import MovingAverageStrategy
from src.strategies.hy_timing_strategy import HYTimingStrategy
from src.strategies.mras_strategy import MRASStrategy

###########################################
# Raw Stats Functions (from raw_stats.py)
###########################################

def load_data(file_path):
    """Load and preprocess data."""
    try:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Convert 'Date' column to datetime and set as index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            # Remove duplicates
            df = df[~df.index.duplicated(keep='last')]
            
            # Fill missing values
            df.fillna(method='ffill', inplace=True)
            
            print("\nSuccessfully loaded data with shape:", df.shape)
            print(f"Date range: {df.index.min()} to {df.index.max()}")
            
            return df
        else:
            print("Error: 'Date' column not found in the CSV file.")
            return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def get_benchmark_stats(df, target_column, config, output_dir):
    """Get raw VectorBT stats for the benchmark (buy and hold)."""
    print("\n=== Calculating Buy & Hold Benchmark Stats ===")
    
    # Get price series
    price_series = df[target_column]
    
    # Detect frequency
    freq = pd.infer_freq(df.index)
    print(f"Detected frequency: {freq}")
    
    # Create portfolio
    portfolio = vbt.Portfolio.from_holding(
        price_series,
        init_cash=config['backtest_settings']['initial_capital'],
        size=config['backtest_settings']['size'],
        size_type=config['backtest_settings']['size_type'],
        freq=freq if freq is not None else '1D'  # Default to daily if frequency cannot be inferred
    )
    
    # Save portfolio stats
    stats = portfolio.stats()
    returns_stats = portfolio.returns_acc.stats()
    
    print("\nRaw Portfolio Stats for Buy & Hold:")
    print("="*80)
    print(stats)
    
    print("\nRaw Returns Stats for Buy & Hold:")
    print("="*80)
    print(returns_stats)
    
    # Save stats to file
    stats_file = os.path.join(output_dir, 'benchmark_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Raw Portfolio Stats for Buy & Hold:\n")
        f.write("="*80 + "\n")
        f.write(stats.to_string())
        f.write("\n\nRaw Returns Stats for Buy & Hold:\n")
        f.write("="*80 + "\n")
        f.write(returns_stats.to_string())
    
    return portfolio, stats, returns_stats, price_series, freq if freq is not None else '1D'

def get_ma_strategy_stats(df, price_series, config, freq, output_dir):
    """Get raw VectorBT stats for the MA strategy."""
    print("\n=== Calculating Moving Average Strategy Stats ===")
    
    try:
        # Get MA window from config
        ma_window = config['strategies']['MA']['ma_window']
        
        # Calculate moving average
        ma = price_series.rolling(window=ma_window).mean()
        
        # Generate signals: buy when price > MA, sell when price < MA
        signals = price_series > ma
        
        # Debug info
        print(f"Original signal count: {len(signals)}")
        print(f"Original signal changes: {(signals != signals.shift(1)).sum()}")
        
        # If rebalance frequency is monthly, resample signals to monthly frequency
        if config['backtest_settings']['rebalance_freq'] == 'M':
            print(f"Applying monthly rebalancing for MA strategy")
            
            # Create a new DataFrame with Date as index
            signals_df = pd.DataFrame({'signal': signals})
            signals_df.index.name = 'Date'
            
            # Print the first few rows to debug
            print(f"Original signals head:\n{signals_df.head()}")
            
            # Resample to monthly frequency and take the last signal of each month
            monthly_signals = signals_df.resample('M').last()
            print(f"Monthly signals count: {len(monthly_signals)}")
            print(f"Monthly signals head:\n{monthly_signals.head()}")
            
            # Forward fill the monthly signals to get daily signals
            resampled_signals = monthly_signals.reindex(signals_df.index, method='ffill')
            print(f"Resampled signals count: {len(resampled_signals)}")
            print(f"Resampled signals head:\n{resampled_signals.head()}")
            
            # Convert back to Series
            signals = resampled_signals['signal']
            
            # Debug info
            print(f"Modified signal count: {len(signals)}")
            print(f"Modified signal changes: {(signals != signals.shift(1)).sum()}")
        
        # Generate entries and exits
        entries = signals & ~signals.shift(1).fillna(False)
        exits = ~signals & signals.shift(1).fillna(False)
        
        # Debug info
        print(f"Number of entries: {entries.sum()}")
        print(f"Number of exits: {exits.sum()}")
        
        # Create portfolio
        ma_portfolio = vbt.Portfolio.from_signals(
            price_series,
            entries=entries,
            exits=exits,
            init_cash=config['backtest_settings']['initial_capital'],
            size=config['backtest_settings']['size'],
            size_type=config['backtest_settings']['size_type'],
            freq=freq if freq is not None else '1D'
        )
        
        # Get portfolio stats
        ma_stats = ma_portfolio.stats()
        ma_returns_stats = ma_portfolio.returns_acc.stats()
        
        print("\nRaw Portfolio Stats for MA Strategy:")
        print("="*80)
        print(ma_stats)
        
        print("\nRaw Returns Stats for MA Strategy:")
        print("="*80)
        print(ma_returns_stats)
        
        # Save stats to files
        with open(os.path.join(output_dir, "ma_portfolio_stats.txt"), 'w') as f:
            f.write(str(ma_stats))
        
        with open(os.path.join(output_dir, "ma_returns_stats.txt"), 'w') as f:
            f.write(str(ma_returns_stats))
        
        return ma_portfolio, ma_stats, ma_returns_stats
    except Exception as e:
        print(f"Error calculating MA strategy stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def get_hy_strategy_stats(df, price_series, config, freq, output_dir):
    """Get raw VectorBT stats for the HY Timing strategy."""
    print("\n=== Calculating HY Timing Strategy Stats ===")
    
    try:
        # Get HY MA window from config
        hy_ma_window = config['strategies']['HYTiming']['ma_window']
        
        # Get HY index column
        hy_column = 'us_hy_er_index'
        
        # Calculate moving average of HY index
        hy_ma = df[hy_column].rolling(window=hy_ma_window).mean()
        
        # Generate signals: buy when HY > MA, sell when HY < MA
        signals = df[hy_column] > hy_ma
        
        # Debug info
        print(f"Original signal count: {len(signals)}")
        print(f"Original signal changes: {(signals != signals.shift(1)).sum()}")
        
        # If rebalance frequency is monthly, resample signals to monthly frequency
        if config['backtest_settings']['rebalance_freq'] == 'M':
            print(f"Applying monthly rebalancing for HY Timing strategy")
            
            # Create a new DataFrame with Date as index
            signals_df = pd.DataFrame({'signal': signals})
            signals_df.index.name = 'Date'
            
            # Print the first few rows to debug
            print(f"Original signals head:\n{signals_df.head()}")
            
            # Resample to monthly frequency and take the last signal of each month
            monthly_signals = signals_df.resample('M').last()
            print(f"Monthly signals count: {len(monthly_signals)}")
            print(f"Monthly signals head:\n{monthly_signals.head()}")
            
            # Forward fill the monthly signals to get daily signals
            resampled_signals = monthly_signals.reindex(signals_df.index, method='ffill')
            print(f"Resampled signals count: {len(resampled_signals)}")
            print(f"Resampled signals head:\n{resampled_signals.head()}")
            
            # Convert back to Series
            signals = resampled_signals['signal']
            
            # Debug info
            print(f"Modified signal count: {len(signals)}")
            print(f"Modified signal changes: {(signals != signals.shift(1)).sum()}")
        
        # Generate entries and exits
        entries = signals & ~signals.shift(1).fillna(False)
        exits = ~signals & signals.shift(1).fillna(False)
        
        # Debug info
        print(f"Number of entries: {entries.sum()}")
        print(f"Number of exits: {exits.sum()}")
        
        # Create portfolio
        hy_portfolio = vbt.Portfolio.from_signals(
            price_series,
            entries=entries,
            exits=exits,
            init_cash=config['backtest_settings']['initial_capital'],
            size=config['backtest_settings']['size'],
            size_type=config['backtest_settings']['size_type'],
            freq=freq if freq is not None else '1D'
        )
        
        # Get portfolio stats
        hy_stats = hy_portfolio.stats()
        hy_returns_stats = hy_portfolio.returns_acc.stats()
        
        print("\nRaw Portfolio Stats for HY Timing Strategy:")
        print("="*80)
        print(hy_stats)
        
        print("\nRaw Returns Stats for HY Timing Strategy:")
        print("="*80)
        print(hy_returns_stats)
        
        # Save stats to files
        with open(os.path.join(output_dir, "hy_portfolio_stats.txt"), 'w') as f:
            f.write(str(hy_stats))
        
        with open(os.path.join(output_dir, "hy_returns_stats.txt"), 'w') as f:
            f.write(str(hy_returns_stats))
        
        return hy_portfolio, hy_stats, hy_returns_stats
    except Exception as e:
        print(f"Error calculating HY Timing strategy stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def get_mras_strategy_stats(df, price_series, config, freq, output_dir):
    """Get raw VectorBT stats for the MRAS strategy."""
    print("\n=== Calculating MRAS Strategy Stats ===")
    
    try:
        # Create MRAS strategy instance and generate signals
        mras_strategy = MRASStrategy(config)  # Pass only the config
        
        # The MRAS strategy expects a DataFrame with multiple columns, not just a price series
        signals = mras_strategy.generate_signals(df)
        
        # Convert signals to boolean for vectorbt
        signals_bool = signals > 0
        
        # Debug info
        print(f"Original signal count: {len(signals_bool)}")
        print(f"Original signal changes: {(signals_bool != signals_bool.shift(1)).sum()}")
        
        # If rebalance frequency is monthly, resample signals to monthly frequency
        if config['backtest_settings']['rebalance_freq'] == 'M':
            print(f"Applying monthly rebalancing for MRAS strategy")
            
            # Create a new DataFrame with Date as index
            signals_df = pd.DataFrame({'signal': signals_bool})
            signals_df.index.name = 'Date'
            
            # Print the first few rows to debug
            print(f"Original signals head:\n{signals_df.head()}")
            
            # Resample to monthly frequency and take the last signal of each month
            monthly_signals = signals_df.resample('M').last()
            print(f"Monthly signals count: {len(monthly_signals)}")
            print(f"Monthly signals head:\n{monthly_signals.head()}")
            
            # Forward fill the monthly signals to get daily signals
            resampled_signals = monthly_signals.reindex(signals_df.index, method='ffill')
            print(f"Resampled signals count: {len(resampled_signals)}")
            print(f"Resampled signals head:\n{resampled_signals.head()}")
            
            # Convert back to Series
            signals_bool = resampled_signals['signal']
            
            # Debug info
            print(f"Modified signal count: {len(signals_bool)}")
            print(f"Modified signal changes: {(signals_bool != signals_bool.shift(1)).sum()}")
        
        # Generate entries and exits
        entries = signals_bool & ~signals_bool.shift(1).fillna(False)
        exits = ~signals_bool & signals_bool.shift(1).fillna(False)
        
        # Debug info
        print(f"Number of entries: {entries.sum()}")
        print(f"Number of exits: {exits.sum()}")
        
        # Create portfolio
        mras_portfolio = vbt.Portfolio.from_signals(
            price_series,
            entries=entries,
            exits=exits,
            init_cash=config['backtest_settings']['initial_capital'],
            size=config['backtest_settings']['size'],
            size_type=config['backtest_settings']['size_type'],
            freq=freq if freq is not None else '1D'
        )
        
        # Get portfolio stats
        mras_stats = mras_portfolio.stats()
        mras_returns_stats = mras_portfolio.returns_acc.stats()
        
        print("\nRaw Portfolio Stats for MRAS Strategy:")
        print("="*80)
        print(mras_stats)
        
        print("\nRaw Returns Stats for MRAS Strategy:")
        print("="*80)
        print(mras_returns_stats)
        
        # Save stats to file
        stats_file = os.path.join(output_dir, 'mras_strategy_stats.txt')
        with open(stats_file, 'w') as f:
            f.write("Raw Portfolio Stats for MRAS Strategy:\n")
            f.write("="*80 + "\n")
            f.write(mras_stats.to_string())
            f.write("\n\nRaw Returns Stats for MRAS Strategy:\n")
            f.write("="*80 + "\n")
            f.write(mras_returns_stats.to_string())
        
        return mras_portfolio, mras_stats, mras_returns_stats
        
    except Exception as e:
        print(f"Error in MRAS strategy: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_comparison_table(benchmark_stats, ma_stats, hy_stats, mras_stats):
    """Create a comparison table of key metrics."""
    # Define metrics to compare - make sure these match the actual keys in the stats objects
    metrics = {
        'Total Return [%]': lambda x: x.get('Total Return [%]', None) if x is not None else None,
        'Annualized Return [%]': lambda x: x.get('Annualized Return [%]', None) if x is not None else None,
        'Annualized Volatility [%]': lambda x: x.get('Annualized Volatility [%]', None) if x is not None else None,
        'Sharpe Ratio': lambda x: x.get('Sharpe Ratio', None) if x is not None else None,
        'Max Drawdown [%]': lambda x: x.get('Max Drawdown [%]', None) if x is not None else None,
        'Calmar Ratio': lambda x: x.get('Calmar Ratio', None) if x is not None else None,
        'Sortino Ratio': lambda x: x.get('Sortino Ratio', None) if x is not None else None
    }
    
    # Create comparison table
    comparison = {
        'Benchmark': [metric(benchmark_stats) for metric in metrics.values()],
        'MA Strategy': [metric(ma_stats) for metric in metrics.values()],
        'HY Timing': [metric(hy_stats) for metric in metrics.values()]
    }
    
    # Add MRAS strategy if available
    if mras_stats is not None:
        comparison['MRAS'] = [metric(mras_stats) for metric in metrics.values()]
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison, index=metrics.keys())
    return comparison_df

def run_comparison(output_dir, benchmark_returns, ma_returns, hy_returns, mras_returns, 
                  benchmark_stats, ma_stats, hy_stats, mras_stats):
    """Run the strategy comparison."""
    try:
        print("\nCreating comparison tables...")
        
        # Create comparison table
        returns_comparison = create_comparison_table(benchmark_returns, ma_returns, hy_returns, mras_returns)
        stats_comparison = create_comparison_table(benchmark_stats, ma_stats, hy_stats, mras_stats)
        
        # Print comparison tables
        print("\nReturns Comparison:")
        print("="*80)
        print(returns_comparison)
        
        print("\nStats Comparison:")
        print("="*80)
        print(stats_comparison)
        
        # Save comparison tables
        returns_comparison.to_csv(os.path.join(output_dir, 'returns_comparison.csv'))
        stats_comparison.to_csv(os.path.join(output_dir, 'stats_comparison.csv'))
        
        # Create visualizations
        print("\nCreating visualizations...")
        
        # Save comparison tables as HTML
        returns_comparison.to_html(os.path.join(output_dir, 'returns_comparison.html'))
        stats_comparison.to_html(os.path.join(output_dir, 'stats_comparison.html'))
        
        print("\nAll comparisons saved to", output_dir)
        
    except Exception as e:
        print(f"Error in comparison: {str(e)}")
        import traceback
        traceback.print_exc()

def run_raw_stats(project_root, config):
    """Run the raw stats calculations."""
    print("\n" + "="*80)
    print("STEP 1: Running raw_stats.py to generate strategy statistics")
    print("="*80 + "\n")
    
    # Load data
    data_path = os.path.join(project_root, config['data']['file_path'])
    df = load_data(data_path)
    
    if df is None:
        print("Error: Failed to load data.")
        return None
    
    # Get target column
    target_column = config['data']['target_column']
    print(f"\nAnalyzing target column: {target_column}")
    
    # Set up output directory
    output_dir = os.path.join(project_root, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate stats for each strategy
    benchmark_portfolio, benchmark_stats, benchmark_returns, price_series, freq = get_benchmark_stats(df, target_column, config, output_dir)
    ma_portfolio, ma_stats, ma_returns = get_ma_strategy_stats(df, price_series, config, freq, output_dir)
    hy_portfolio, hy_stats, hy_returns = get_hy_strategy_stats(df, price_series, config, freq, output_dir)
    mras_portfolio, mras_stats, mras_returns = get_mras_strategy_stats(df, price_series, config, freq, output_dir)
    
    print("\nSaved all stats to", output_dir)
    
    return (benchmark_portfolio, ma_portfolio, hy_portfolio, mras_portfolio,
            benchmark_stats, ma_stats, hy_stats, mras_stats,
            benchmark_returns, ma_returns, hy_returns, mras_returns)

def main():
    """Main function to run and analyze strategies."""
    try:
        # Get project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        
        # Load config
        config_path = os.path.join(project_root, 'config', 'backtest_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print("\n" + "="*80)
        print("BACKTESTING FRAMEWORK INTEGRATED SYSTEM")
        print("="*80)
        
        print("\nThis script integrates the raw_stats.py and strategy_comparison.py functionality")
        print("It demonstrates the relationship between all components of the system\n")
        
        print("="*80)
        print("STEP 1: Running raw_stats.py to generate strategy statistics")
        print("="*80)
        
        print("\nExecuting raw_stats.py...")
        
        # Run raw stats
        results = run_raw_stats(project_root, config)
        if results is None:
            print("Error: Failed to run raw statistics.")
            return
            
        (benchmark_portfolio, ma_portfolio, hy_portfolio, mras_portfolio,
         benchmark_stats, ma_stats, hy_stats, mras_stats,
         benchmark_returns, ma_returns, hy_returns, mras_returns) = results
        
        print("\n" + "="*80)
        print("STEP 2: Running strategy_comparison.py to compare strategies")
        print("="*80)
        
        print("\nExecuting strategy_comparison.py...")
        
        run_comparison(os.path.join(project_root, 'results'), 
                      benchmark_returns, ma_returns, hy_returns, mras_returns,
                      benchmark_stats, ma_stats, hy_stats, mras_stats)
        
        print("\n" + "="*80)
        print("BACKTESTING COMPLETE")
        print("="*80)
        
        print("\nAll statistics and comparisons have been generated in the /results directory")
        print("\nThe integration demonstrates how the whole system works together:")
        print("1. raw_stats.py - Runs strategies and generates raw statistics")
        print("2. strategy_comparison.py - Compares all strategies and creates visualizations")
        print("3. run_strategies.py - Orchestrates the overall system")
        
        # Save all output to integrated_output.txt
        output_file = os.path.join(project_root, 'results', 'integrated_output.txt')
        with open(output_file, 'w') as f:
            # Re-run everything to capture output
            # This is a bit inefficient but ensures we get clean output
            import io
            from contextlib import redirect_stdout
            
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                print("\n" + "="*80)
                print("BACKTESTING FRAMEWORK INTEGRATED SYSTEM")
                print("="*80)
                
                print("\nThis script integrates the raw_stats.py and strategy_comparison.py functionality")
                print("It demonstrates the relationship between all components of the system\n")
                
                print("="*80)
                print("STEP 1: Running raw_stats.py to generate strategy statistics")
                print("="*80)
                
                # Re-run raw stats for output capture
                run_raw_stats(project_root, config)
                
                print("\n" + "="*80)
                print("STEP 2: Running strategy_comparison.py to compare strategies")
                print("="*80)
                
                run_comparison(os.path.join(project_root, 'results'), 
                              benchmark_returns, ma_returns, hy_returns, mras_returns,
                              benchmark_stats, ma_stats, hy_stats, mras_stats)
                
                print("\n" + "="*80)
                print("BACKTESTING COMPLETE")
                print("="*80)
                
                print("\nAll statistics and comparisons have been generated in the /results directory")
                print("\nThe integration demonstrates how the whole system works together:")
                print("1. raw_stats.py - Runs strategies and generates raw statistics")
                print("2. strategy_comparison.py - Compares all strategies and creates visualizations")
                print("3. run_strategies.py - Orchestrates the overall system")
            
            # Write the captured output to the file
            f.write(buffer.getvalue())
        
    except Exception as e:
        import traceback
        print(f"\nError running strategies: {str(e)}")
        traceback.print_exc()
        
if __name__ == "__main__":
    main()
