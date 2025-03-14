from datetime import datetime
import pandas as pd
import os
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.bloomberg_fetcher import fetch_bloomberg_data
from src.utils.transformations import convert_er_ytd_to_index
from src.utils.data_merger import merge_dfs

def fetch_market_data(start_date='2002-01-01', end_date=None):
    """
    Fetches and processes market data from Bloomberg.
    
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format. Defaults to '2002-01-01'
        end_date (str): End date in 'YYYY-MM-DD' format. Defaults to current date
        
    Returns:
        pd.DataFrame: Processed market data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # Market data mapping
    market_data_mapping = {
        ('I05510CA Index', 'INDEX_OAS_TSY_BP'): 'cad_oas',  # CAD IG OAS
        ('LF98TRUU Index', 'INDEX_OAS_TSY_BP'): 'us_hy_oas',  # US HY OAS
        ('LUACTRUU Index', 'INDEX_OAS_TSY_BP'): 'us_ig_oas',  # US IG OAS
        ('SPTSX Index', 'PX_LAST'): 'tsx',  # TSX Index
        ('VIX Index', 'PX_LAST'): 'vix',  # VIX Index
        ('USYC3M30 Index', 'PX_LAST'): 'us_3m_10y',  # US 3M-30Y Slope
        ('BCMPUSGR Index', 'PX_LAST'): 'us_growth_surprises',  # US Growth Surprises
        ('BCMPUSIF Index', 'PX_LAST'): 'us_inflation_surprises',  # US Inflation Surprises
        ('LEI YOY  Index', 'PX_LAST'): 'us_lei_yoy',  # US LEI YoY
        ('.HARDATA G Index', 'PX_LAST'): 'us_hard_data_surprises',  # US Hard Data Surprises
        ('.ECONREGI G Index', 'PX_LAST'): 'us_economic_regime',  # US Economic Regime
    }

    # Excess return YTD mapping
    er_ytd_mapping = {
        ('I05510CA Index', 'INDEX_EXCESS_RETURN_YTD'): 'cad_ig_er',
        ('LF98TRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_hy_er',
        ('LUACTRUU Index', 'INDEX_EXCESS_RETURN_YTD'): 'us_ig_er',
    }

    # Fetch market data
    market_df = fetch_bloomberg_data(
        mapping=market_data_mapping,
        start_date=start_date,
        end_date=end_date,
        periodicity='M',  # Changed from 'D' to 'M' for monthly data
        align_start=True
    ).dropna()

    # Fetch excess return data
    er_df = fetch_bloomberg_data(
        mapping=er_ytd_mapping,
        start_date=start_date,
        end_date=end_date,
        periodicity='M',  # Changed from 'D' to 'M' for monthly data
        align_start=True
    ).dropna()

    # Convert excess return YTD data to index
    er_index_df = convert_er_ytd_to_index(
        er_df[['cad_ig_er', 'us_hy_er', 'us_ig_er']]
    )

    # Merge dataframes
    final_df = merge_dfs(
        market_df, 
        er_index_df, 
        fill='ffill', 
        start_date_align='yes'
    )

    # Handle bad data point for cad_oas on Nov 15 2005
    bad_date = '2005-11-15'
    if bad_date in final_df.index:
        final_df.loc[bad_date, 'cad_oas'] = final_df.loc[
            final_df.index < bad_date, 'cad_oas'
        ].iloc[-1]

    return final_df

if __name__ == '__main__':
    # Fetch the data
    data = fetch_market_data()
    print("\nDataset Info:")
    print(data.info())
    
    # Export to CSV - directly overwrite if exists
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data.csv')
    data.to_csv(output_file)
    
    print(f"\nFirst few rows of the data:")
    print(data.head())
    print(f"\nData has been exported to: {output_file}")

