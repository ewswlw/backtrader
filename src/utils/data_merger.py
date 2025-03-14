"""
Utility functions for merging DataFrames with special handling for missing values and date alignment.
"""
import pandas as pd
import logging
from typing import Union, List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def merge_dfs(*dfs_args: Union[pd.DataFrame, pd.Series], 
              fill: str = None, 
              start_date_align: str = "no") -> pd.DataFrame:
    """
    Merge multiple DataFrames or Series with options for handling missing values and date alignment.
    
    Args:
        *dfs_args: Variable number of DataFrames/Series to merge
        fill (str, optional): Fill method for NaN values. Options: None, 'ffill', 'bfill', 'interpolate'
        start_date_align (str, optional): Whether to align start dates. Options: 'yes', 'no'
        
    Returns:
        pd.DataFrame: Merged DataFrame with date index
    """
    # Convert arguments to dictionary
    dfs = {f'df_{i}': df for i, df in enumerate(dfs_args)}
    
    # Convert Series to DataFrame and ensure datetime index
    processed_dfs = {}
    for name, df in dfs.items():
        # Convert Series to DataFrame if necessary
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(f"Failed to convert index to datetime for {name}: {str(e)}")
        
        processed_dfs[name] = df
    
    # Find first valid date for each DataFrame
    first_valid_dates = {}
    for name, df in processed_dfs.items():
        for col in df.columns:
            first_valid_idx = df[col].first_valid_index()
            if first_valid_idx is not None:
                if name not in first_valid_dates:
                    first_valid_dates[name] = first_valid_idx
                else:
                    first_valid_dates[name] = max(first_valid_dates[name], first_valid_idx)
    
    # Align start dates if requested
    start_date_align = str(start_date_align).lower()
    if start_date_align == "yes" and first_valid_dates:
        # Find the latest start date among all DataFrames
        latest_start = max(first_valid_dates.values())
        
        # Trim all DataFrames to start from this date
        for name in processed_dfs:
            df = processed_dfs[name]
            mask = df.index >= latest_start
            processed_dfs[name] = df.loc[mask].copy()
    
    # Merge all DataFrames
    merged_df = None
    for name, df in processed_dfs.items():
        if merged_df is None:
            merged_df = df.copy()
        else:
            # Merge with outer join to keep all dates
            merged_df = pd.merge(merged_df, df, 
                               left_index=True, right_index=True, 
                               how='outer')
    
    if merged_df is None:
        return pd.DataFrame()
    
    # Apply fill method if specified
    if fill is not None:
        if fill == 'ffill':
            # Forward fill but only after first valid value for each column
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].ffill()
        
        elif fill == 'bfill':
            # Backward fill but only after first valid value for each column
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].bfill()
        
        elif fill == 'interpolate':
            # Interpolate but only after first valid value for each column
            for col in merged_df.columns:
                first_valid_idx = merged_df[col].first_valid_index()
                if first_valid_idx is not None:
                    mask = merged_df.index >= first_valid_idx
                    merged_df.loc[mask, col] = merged_df.loc[mask, col].interpolate(method='time')
    
    return merged_df
