import pandas as pd
import numpy as np

def convert_er_ytd_to_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert excess return YTD data to an index
    
    Args:
        df: DataFrame containing excess return YTD columns
        
    Returns:
        DataFrame with excess return columns converted to indices
    """
    result = pd.DataFrame(index=df.index)
    
    for column in df.columns:
        # Convert YTD returns to daily returns
        daily_returns = df[column].diff()
        
        # Create index starting at 100
        index_values = (1 + daily_returns/100).cumprod() * 100
        result[f"{column}_index"] = index_values
        
    return result
