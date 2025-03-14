import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Union
from xbbg import blp

def fetch_bloomberg_data(
    mapping: Dict[Tuple[str, str], str],
    start_date: str,
    end_date: str,
    periodicity: str = 'D',
    align_start: bool = True
) -> pd.DataFrame:
    """
    Fetch data from Bloomberg using xbbg
    
    Args:
        mapping: Dictionary mapping (security, field) tuples to column names
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        periodicity: Data frequency ('D' for daily)
        align_start: Whether to align data from the start date
        
    Returns:
        DataFrame with requested data
    """
    securities = list(set(security for security, _ in mapping.keys()))
    fields = list(set(field for _, field in mapping.keys()))

    # Fetch data using xbbg
    df = blp.bdh(
        tickers=securities,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
        Per=periodicity
    )

    # Create a new DataFrame with renamed columns
    renamed_df = pd.DataFrame(index=df.index)
    for (security, field), new_name in mapping.items():
        if (security, field) in df.columns:
            renamed_df[new_name] = df[(security, field)]
    
    return renamed_df
