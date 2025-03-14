import pandas as pd
from xbbg import blp
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

def fetch_bloomberg_data(mapping, start_date='2000-01-01', end_date=None, periodicity='D', align_start=False):
    """
    Fetch Bloomberg data for multiple securities and fields.
    
    Args:
        mapping (dict): Dictionary mapping (ticker, field) tuples to column names
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format. Defaults to today.
        periodicity (str): Data frequency ('D' for daily, 'M' for monthly, etc.)
        align_start (bool): If True, align all series to start from the latest first valid date
    
    Returns:
        pd.DataFrame: DataFrame with datetime index and requested data columns
    """
    if end_date is None:
        end_date = pd.Timestamp('today').strftime('%Y-%m-%d')
    
    tickers = list({k[0] for k in mapping.keys()})
    fields = list({k[1] for k in mapping.keys()})
    
    df_raw = blp.bdh(
        tickers=tickers,
        flds=fields,
        start_date=start_date,
        end_date=end_date,
        Per=periodicity
    )
    
    # Ensure index is datetime
    df_raw.index = pd.to_datetime(df_raw.index)
    
    df_raw.columns = df_raw.columns.to_flat_index()
    df_raw.rename(columns=lambda col: mapping.get(col, f"{col[0]}|{col[1]}"), inplace=True)
    
    desired_order = [mapping[pair] for pair in mapping]
    final_df = df_raw[[c for c in desired_order if c in df_raw.columns]]
    
    if align_start:
        first_valid_per_col = []
        for col in final_df.columns:
            first_valid = final_df[col].first_valid_index()
            if first_valid is not None:
                first_valid_per_col.append(first_valid)
        if first_valid_per_col:
            start_cutoff = max(first_valid_per_col)
            final_df = final_df.loc[final_df.index >= start_cutoff]
    
    # Make a copy here before forward-fill
    final_df = final_df.copy()
    
    # Forward-fill
    final_df = final_df.ffill()
    
    # Drop duplicates, sort index, etc.
    final_df = final_df.loc[~final_df.index.duplicated(keep='first')].copy()
    final_df.sort_index(inplace=True)
    final_df.index.name = 'Date'
    
    # Final check to ensure datetime index
    if not isinstance(final_df.index, pd.DatetimeIndex):
        final_df.index = pd.to_datetime(final_df.index)
    
    return final_df

class BloombergDataFetcher:
    """Class to fetch and process Bloomberg data."""
    
    def __init__(
        self,
        tickers: List[str],
        fields: List[str],
        start_date: str,
        end_date: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        config: Optional[Dict] = None
    ):
        self.tickers = tickers
        self.fields = fields
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today()
        self.logger = logger or logging.getLogger(__name__)
        self.custom_names = {}
        self.config = config or {}
        
    def set_custom_column_names(self, custom_names: Dict[str, str]):
        """Set custom names for renaming columns."""
        self.custom_names = custom_names
        
    def fetch_data(self) -> pd.DataFrame:
        """Fetch data from Bloomberg."""
        try:
            all_data = []
            self.logger.info(f"Starting data fetch with tickers: {self.tickers} and fields: {self.fields}")
            
            # If using direct ticker/field approach
            if self.tickers and self.fields:
                for ticker in self.tickers:
                    for field in self.fields:
                        self.logger.info(f"Fetching data for ticker: {ticker}, field: {field}")
                        data = blp.bdh(
                            ticker,
                            field,
                            self.start_date,
                            self.end_date
                        )
                        
                        if not data.empty:
                            # Get the custom name for this ticker if available
                            custom_name = self.custom_names.get(ticker, ticker)
                            self.logger.info(f"Data fetched successfully for {ticker}, using custom name: {custom_name}")
                            
                            # Extract the data values and create a new DataFrame with the custom name
                            if isinstance(data.columns, pd.MultiIndex):
                                values = data.iloc[:, 0].values  # Get first column's values
                                data = pd.DataFrame(values, index=data.index, columns=[custom_name])
                            else:
                                data.columns = [custom_name]
                            
                            # Remove any rows with NaN values
                            data = data.dropna()
                            all_data.append(data)
                        else:
                            self.logger.warning(f"Empty data returned for ticker: {ticker}, field: {field}")
            
            # If using config-based approach
            elif self.config:
                for data_type, config in self.config.items():
                    if data_type not in ['sprds', 'derv', 'settings']:
                        continue
                        
                    if data_type in ['sprds', 'derv']:
                        field = config['field']
                        for security in config['securities']:
                            ticker = security['ticker']
                            custom_name = security['custom_name']
                            
                            self.logger.info(f"Fetching data for ticker: {ticker}, field: {field}")
                            data = blp.bdh(
                                ticker,
                                field,
                                self.start_date,
                                self.end_date
                            )
                            
                            if not data.empty:
                                self.logger.info(f"Data fetched successfully for {ticker}, using custom name: {custom_name}")
                                
                                # Extract the data values and create a new DataFrame with the custom name
                                if isinstance(data.columns, pd.MultiIndex):
                                    values = data.iloc[:, 0].values  # Get first column's values
                                    data = pd.DataFrame(values, index=data.index, columns=[custom_name])
                                else:
                                    data.columns = [custom_name]
                                
                                # Remove any rows with NaN values
                                data = data.dropna()
                                all_data.append(data)
                            else:
                                self.logger.warning(f"Empty data returned for ticker: {ticker}, field: {field}")
            
            if not all_data:
                self.logger.error("No data was fetched from Bloomberg")
                raise ValueError("No data fetched from Bloomberg")
            
            # Combine all data
            combined_data = pd.concat(all_data, axis=1)
            self.logger.info(f"Successfully combined data with columns: {combined_data.columns.tolist()}")
            
            # Remove any rows where any values are NaN
            combined_data = combined_data.dropna(how='any')
            
            # Sort index
            combined_data = combined_data.sort_index()
            
            return combined_data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise

    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the fetched data."""
        try:
            # Remove future dates
            today = pd.Timestamp.today().normalize()  # Get today's date at midnight
            data = data[data.index.to_series().apply(lambda x: pd.Timestamp(x) <= today)]
            
            # Remove any rows where any values are NaN
            data = data.dropna(how='any')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing data: {str(e)}")
            raise
            
    def print_data_validation(self, data: pd.DataFrame, title: str):
        """Print comprehensive data validation information."""
        print(f"\n{'='*50}")
        print(f"Data Validation for: {title}")
        print(f"{'='*50}\n")

        # Print DataFrame info
        print(f"{title} Data Info:")
        print("----------------------------------")
        print(data.info())
        print()

        # Print date range and missing dates
        print("Date Range:", data.index.min(), "to", data.index.max())
        business_days = pd.date_range(start=data.index.min(), end=data.index.max(), freq='B')
        missing_days = business_days.difference(data.index)
        if len(missing_days) > 0:
            print(f"WARNING: Found {len(missing_days)} missing business days")
            print("First few missing dates:", missing_days[:5].tolist())
        print()

        # Print column data types
        print("Column Data Types:")
        for col, dtype in data.dtypes.items():
            print(f"  {col}: {dtype}")
        print()

        # Print value range validation
        print("Value Range Validation:\n")
        print("Basic Statistics:\n")
        for column in data.columns:
            stats = data[column].describe()
            print(f"{column}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Std: {stats['std']:.2f}")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print()

        # Print first and last rows
        print("\nFirst 5 rows:")
        print("----------------------------------")
        print(data.head())
        print("\nLast 5 rows:")
        print("----------------------------------")
        print(data.tail())
        print()

    def run_pipeline(self) -> pd.DataFrame:
        """Run the complete data fetching and processing pipeline."""
        self.logger.info("Starting data pipeline")
        self.logger.info(f"Using tickers: {self.tickers}")
        self.logger.info(f"Using fields: {self.fields}")
        self.logger.info(f"Custom names mapping: {self.custom_names}")
        
        data = self.fetch_data()
        self.logger.info("Data fetched successfully")
        
        # Print validation for raw Bloomberg data
        if self.config:
            for data_type in self.config:
                if data_type in ['sprds', 'derv']:
                    cols = [sec['custom_name'] for sec in self.config[data_type]['securities']]
                    if all(col in data.columns for col in cols):
                        self.print_data_validation(data[cols], f"{data_type.title()} Data")
        else:
            self.print_data_validation(data, "Bloomberg Data")
        
        processed_data = self.process_data(data)
        self.logger.info("Data processed successfully")
        
        return processed_data
