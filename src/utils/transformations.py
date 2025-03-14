import pandas as pd
from typing import List, Optional, Union
import logging
from xbbg import blp
import os
import numpy as np
from pandas.tseries.frequencies import to_offset
import warnings
from typing import Tuple

# Set up logging only if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_frequency(frequency: str) -> str:
    """Normalize frequency input to the expected Bloomberg format."""
    freq_map = {
        'DAILY': 'DAILY', 'D': 'DAILY', '1D': 'DAILY', 'DAY': 'DAILY',
        'WEEKLY': 'WEEKLY', 'W': 'WEEKLY', '1W': 'WEEKLY', 'WEEK': 'WEEKLY',
        'MONTHLY': 'MONTHLY', 'M': 'MONTHLY', '1M': 'MONTHLY', 'MONTH': 'MONTHLY',
        'QUARTERLY': 'QUARTERLY', 'Q': 'QUARTERLY', '1Q': 'QUARTERLY', 'QUARTER': 'QUARTERLY',
        'YEARLY': 'YEARLY', 'Y': 'YEARLY', '1Y': 'YEARLY', 'YEAR': 'YEARLY',
        'BM': 'MONTHLY', 'BME': 'MONTHLY', 'BMONTHLY': 'MONTHLY',  # Treat business month-end as monthly
        'ME': 'MONTHLY'  # Treat month-end as monthly
    }
    normalized_freq = freq_map.get(frequency.upper().strip(), 'DAILY')  # Default to 'DAILY' if not found
    return normalized_freq

def risk_free_index(df: pd.DataFrame, col_name: str = "risk_free") -> pd.DataFrame:
    """
    Computes an index starting at 100 that mimics the compounded effect of investing in the index with annual compounding.
    
    :param df: DataFrame with datetime index and interest rates in percentage format (e.g., 4.65 means 4.65%).
    :param col_name: Optional string to set the name of the compounded index column. Default is 'risk_free'.
    :return: DataFrame with the compounded index.
    """
    # Frequency conversion factors
    frequency_factors = {
        'D': 365,     # Daily
        'B': 252,     # Business day
        'M': 12,      # Monthly
        'Q': 4,       # Quarterly
        'A': 1,       # Annual
        'ME': 12      # Month-End, treated as monthly
    }
    
    # Infer the frequency of the DataFrame
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq is None:
        # Attempt to resample to the most likely frequency (daily) to fill gaps
        logging.warning("Cannot infer frequency. Attempting to resample to daily frequency.")
        df = df.resample('D').ffill()
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            raise ValueError("Cannot infer frequency of the DataFrame index after resampling.")
    
    logging.info(f"Inferred frequency: {inferred_freq}")
    
    # Map the inferred frequency to the relevant period per year
    frequency_key = inferred_freq[:2]  # Use the first two characters to capture 'ME' and others
    if frequency_key not in frequency_factors:
        raise ValueError(f"Inferred frequency '{frequency_key}' not supported. Supported frequencies: 'D', 'B', 'M', 'Q', 'A', 'ME'.")
    
    periods_per_year = frequency_factors[frequency_key]
    
    # Flatten multi-index columns if present
    df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]
    
    # Convert interest rates from percentage to decimal
    df = df / 100.0
    
    # Compute the compounded index
    compounded_index = (1 + df / periods_per_year).cumprod()
    
    # Normalize to start at 100
    compounded_index = 100 * compounded_index / compounded_index.iloc[0]
    
    # Rename the column to indicate it is an index
    compounded_index.columns = [col_name + '_index' for col in df.columns]
    
    return compounded_index

def get_credit_fund_estimated_index(df1: pd.DataFrame, df2: pd.DataFrame, col_name: str = "credit_fund_estimated_index") -> pd.DataFrame:
  """
  Computes an estimated credit fund index starting at 100 by summing the periodic returns of the two input DataFrames
  and constructing a single index. This function can handle DataFrames with different date ranges.

  :param df1: First DataFrame with datetime index and price levels (e.g., daily closing prices).
  :param df2: Second DataFrame with datetime index and price levels.
  :param col_name: Optional string to set the name of the compounded index column. Default is 'credit_fund_estimated_index'.
  :return: DataFrame with a single column representing the credit fund estimated index.
  """
  # Combine the indexes of both DataFrames
  combined_index = df1.index.union(df2.index)
  
  # Reindex both DataFrames to the combined index, forward-filling missing values
  df1_reindexed = df1.reindex(combined_index, method='ffill')
  df2_reindexed = df2.reindex(combined_index, method='ffill')
  
  # Calculate the periodic return of each index
  periodic_return1 = df1_reindexed.pct_change().fillna(0)
  periodic_return2 = df2_reindexed.pct_change().fillna(0)
  
  # Sum the returns from both DataFrames
  total_returns = periodic_return1.sum(axis=1) + periodic_return2.sum(axis=1)
  
  # Compute the estimated index
  estimated_index = (1 + total_returns).cumprod()
  
  # Normalize to start at 100
  estimated_index = 100 * estimated_index / estimated_index.iloc[0]
  
  # Convert the result to a DataFrame and name the column
  estimated_index = estimated_index.to_frame(name=col_name)
  
  return estimated_index

def get_er_index(df: pd.DataFrame, cost_of_borrow: float = 40, leverage: float = 1.0, col_name: str = "er_index") -> pd.DataFrame:
    """
    Computes an excess return index starting at 100 that represents the performance of an investment in the index,
    adjusted for the cost of borrowing and leverage.

    :param df: DataFrame with datetime index and price levels (e.g., daily closing prices).
    :param cost_of_borrow: A float expressed in basis points (e.g., 40 means 40 bps, or 0.40% annual compounding interest rate). Default is 40.
    :param leverage: A float that represents the leverage used to create the new index. Default is 1.0.
    :param col_name: Optional string to set the name of the compounded index column. Default is 'er_index'.
    :return: DataFrame with the excess return index.
    """
    # Convert cost_of_borrow from basis points to a decimal (e.g., 40 bps = 0.004)
    cost_of_borrow_decimal = cost_of_borrow / 10000
    
    # Infer the frequency of the DataFrame
    inferred_freq = pd.infer_freq(df.index)
    
    if inferred_freq is None:
        # Attempt to resample to the most likely frequency (daily) to fill gaps
        logging.warning("Cannot infer frequency. Attempting to resample to daily frequency.")
        df = df.resample('D').ffill()
        inferred_freq = pd.infer_freq(df.index)
        if inferred_freq is None:
            raise ValueError("Cannot infer frequency of the DataFrame index after resampling.")
    
    logging.info(f"Inferred frequency: {inferred_freq}")
    
    # Define frequency factors for cost of borrowing calculation
    frequency_factors = {
        # Daily frequencies
        'D': 365,    # Calendar days
        'B': 252,    # Business days
        
        # Weekly frequencies
        'W': 52,     # Weekly
        'W-MON': 52, 'W-TUE': 52, 'W-WED': 52, 'W-THU': 52, 'W-FRI': 52,
        
        # Monthly frequencies
        'M': 12,     # Month end
        'MS': 12,    # Month start
        'BM': 12,    # Business month end
        'BMS': 12,   # Business month start
        'ME': 12,    # Month end
        
        # Quarterly frequencies
        'Q': 4,      # Quarter end
        'QS': 4,     # Quarter start
        'BQ': 4,     # Business quarter end
        'BQS': 4,    # Business quarter start
        
        # Annual frequencies
        'A': 1,      # Year end
        'AS': 1,     # Year start
        'BA': 1,     # Business year end
        'BAS': 1     # Business year start
    }
    
    # Extract base frequency (first character) if not found in mapping
    if inferred_freq not in frequency_factors:
        base_freq = inferred_freq[0]
        if base_freq in ['D', 'W', 'M', 'Q', 'A']:
            frequency_factor = frequency_factors.get(base_freq, 365)  # Default to daily if unknown
        else:
            logging.warning(f"Unknown frequency {inferred_freq}, defaulting to daily (365).")
            frequency_factor = 365
    else:
        frequency_factor = frequency_factors[inferred_freq]
    
    # Calculate periodic cost of borrowing based on frequency
    periodic_cost = cost_of_borrow_decimal / frequency_factor
    
    # Calculate returns for each column
    result_df = pd.DataFrame(index=df.index)
    
    # Create leverage suffix using the actual leverage value
    leverage_suffix = f"_{leverage:.1f}x"
    
    for column in df.columns:
        # Calculate periodic returns based on the inferred frequency
        periodic_returns = df[column].pct_change().fillna(0)
        
        # Apply leverage and subtract borrowing cost
        leveraged_returns = (periodic_returns * leverage) - (periodic_cost * leverage)
        
        # Calculate cumulative index
        cumulative_index = (1 + leveraged_returns).cumprod()
        
        # Normalize to start at 100
        normalized_index = 100 * cumulative_index / cumulative_index.iloc[0]
        
        # Add to result DataFrame with original column name plus leverage suffix
        result_df[f"{column}{leverage_suffix}"] = normalized_index
    
    return result_df

def convert_er_ytd_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts year-to-date excess return data in all columns to custom indices starting at 100.
    Assumes that all columns contain year-to-date excess return data in percentage format.

    Args:
        df (pd.DataFrame): DataFrame containing year-to-date excess return data in percentage format.

    Returns:
        pd.DataFrame: DataFrame with custom indices columns calculated from daily returns.
    """
    try:
        # Initialize result DataFrame with same index as input
        result = pd.DataFrame(index=df.index)
        
        # Process each column
        for column in df.columns:
            # Convert percentage to decimal (e.g., 5.2% -> 0.052)
            ytd_returns = df[column] / 100.0
            
            # Get years in the data
            years = df.index.year.unique()
            
            # Initialize index values
            index_values = pd.Series(index=df.index, dtype=float)
            prev_year_end = 100  # Start with base value of 100
            
            # Calculate index for each year
            for year in sorted(years):
                year_mask = df.index.year == year
                year_data = ytd_returns[year_mask]
                
                if not year_data.empty:
                    # Calculate index values for this year
                    index_values[year_mask] = prev_year_end * (1 + year_data)
                    
                    # Update previous year-end value for next year's calculation
                    prev_year_end = index_values[year_mask].iloc[-1]
            
            # Add the calculated index to result DataFrame
            result[f"{column}_index"] = index_values
        
        return result
        
    except Exception as e:
        logging.error(f"Error in convert_er_ytd_to_index: {str(e)}")
        raise

def get_ohlc(data: Union[pd.Series, pd.DataFrame], frequency: str) -> pd.DataFrame:
    """
    Resample the input time series data to the specified frequency and return OHLC values.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        The input time series data. If a DataFrame, it should have a DateTime index and a single column.
    frequency : str
        The resampling frequency:
        - 'd': Daily (copies values to OHLC)
        - 'w' or 'W': Week ending Friday (uses 'W-FRI')
        - 'm' or 'M': Month end (uses 'ME')
        - 'q' or 'Q': Quarter end December (uses 'Q-DEC')
        - 'y' or 'Y': Year end December (uses 'A-DEC')

    Returns:
    --------
    pd.DataFrame
        A DataFrame with columns ['Open', 'High', 'Low', 'Close'].
    """
    # Ensure the input is a DataFrame
    if isinstance(data, pd.Series):
        data = data.to_frame(name='value')
    elif isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column.")
    else:
        raise TypeError("Input must be a Pandas Series or DataFrame.")

    # Define the resampling rules
    resample_rules = {
        'd': None,     # Special case - no resampling
        'D': None,     # Special case - no resampling
        'w': 'W-FRI',  # Week ending Friday
        'W': 'W-FRI',
        'm': 'ME',     # Month end
        'M': 'ME',
        'q': 'Q-DEC',  # Quarter end (December fiscal year end)
        'Q': 'Q-DEC',
        'y': 'A-DEC',  # Annual end (December fiscal year end)
        'Y': 'A-DEC'
    }

    # Get the resampling rule
    resample_rule = resample_rules.get(frequency)
    if resample_rule is None:
        if frequency.lower() != 'd':
            valid_freqs = "', '".join(['d', 'w', 'W', 'm', 'M', 'q', 'Q', 'y', 'Y'])
            raise ValueError(f"Invalid frequency. Use one of: '{valid_freqs}'")
        
        # Handle daily frequency - just copy values
        ohlc = pd.DataFrame(index=data.index)
        ohlc['Open'] = data.iloc[:, 0]
        ohlc['High'] = data.iloc[:, 0]
        ohlc['Low'] = data.iloc[:, 0]
        ohlc['Close'] = data.iloc[:, 0]
        return ohlc

    # Resample and calculate OHLC for other frequencies
    ohlc = data.resample(resample_rule).agg(['first', 'max', 'min', 'last'])
    ohlc.columns = ['Open', 'High', 'Low', 'Close']

    return ohlc

def rollchg(data: Union[pd.DataFrame, pd.Series], 
            period: str = '1y', 
            change_type: str = 'pct', 
            lag_lead: int = 0) -> pd.DataFrame:
    """
    Calculate rolling changes for time series data.

    Parameters:
    -----------
    data : pd.DataFrame or pd.Series
        Input time series data with datetime index
    period : str, default '1y'
        Period to calculate change over. Format: number + period identifier
        Period identifiers:
        - 'd' for days (e.g., '5d' for 5 days)
        - 'm' for months (e.g., '3m' for 3 months)
        - 'q' for quarters (e.g., '2q' for 2 quarters)
        - 'y' for years (e.g., '1y' for 1 year)
    change_type : str, default 'pct'
        Type of change to calculate: 'pct' for percentage or 'net' for absolute
    lag_lead : int, default 0
        Number of periods to shift the series by:
        - Negative values lag the series (look back)
        - Positive values lead the series (look forward)
        - Zero means no shift

    Returns:
    --------
    pd.DataFrame
        DataFrame with calculated changes and DatetimeIndex
    """
    # Input validation
    if not isinstance(data, (pd.DataFrame, pd.Series)):
        raise TypeError("Input must be a pandas DataFrame or Series")

    # Ensure we have a DataFrame with datetime index
    if isinstance(data, pd.Series):
        df = data.to_frame()
    else:
        df = data.copy()

    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError(f"Could not convert index to datetime: {str(e)}")

    # Parse period string
    if not isinstance(period, str):
        period = str(period)
    
    # Extract number and period identifier
    import re
    match = re.match(r'(\d*\.?\d*)([dmqy])', period.lower())
    if not match:
        raise ValueError("Invalid period format. Use number + identifier (e.g., '1y', '3m', '90d', '2q')")
    
    num, unit = match.groups()
    num = float(num) if num else 1.0

    # Calculate the number of periods based on data frequency
    avg_delta = (df.index[-1] - df.index[0]) / (len(df.index) - 1)
    data_freq_days = avg_delta.days

    # Convert period to number of observations
    if unit == 'd':
        n_periods = int(num / data_freq_days) if data_freq_days > 0 else int(num)
    elif unit == 'm':
        n_periods = int((num * 30) / data_freq_days) if data_freq_days > 0 else int(num * 21)
    elif unit == 'q':
        n_periods = int((num * 90) / data_freq_days) if data_freq_days > 0 else int(num * 63)
    elif unit == 'y':
        n_periods = int((num * 365) / data_freq_days) if data_freq_days > 0 else int(num * 252)

    # Apply lag/lead if specified
    if lag_lead != 0:
        df = df.shift(-lag_lead)  # Negative shift for lag, positive for lead

    # Calculate changes
    if change_type.lower() == 'pct':
        result = df.pct_change(periods=n_periods, fill_method=None)
    else:  # net change
        result = df - df.shift(n_periods)

    return result.dropna()
