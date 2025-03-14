import pandas as pd
import numpy as np

def merge_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    fill: str = 'ffill',
    start_date_align: str = 'yes'
) -> pd.DataFrame:
    """
    Merge two DataFrames with proper date alignment and filling
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        fill: Fill method ('ffill' for forward fill)
        start_date_align: Whether to align start dates ('yes' or 'no')
        
    Returns:
        Merged DataFrame
    """
    # Merge DataFrames
    merged = pd.concat([df1, df2], axis=1)
    
    if start_date_align == 'yes':
        # Get the latest start date
        start_date = max(df1.index.min(), df2.index.min())
        merged = merged[merged.index >= start_date]
    
    # Fill missing values
    if fill == 'ffill':
        merged = merged.fillna(method='ffill')
    
    return merged
