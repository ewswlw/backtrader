import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Callable
import yfinance as yf
import logging

# Set up logging
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_return_series(series: pd.Series) -> pd.Series:
    """
    Calculates the return series of a given time series.
    
    Args:
        series (pd.Series): Price series with datetime index
        
    Returns:
        pd.Series: Return series where first value will be NaN
    """
    shifted_series = series.shift(1, axis=0)
    return series / shifted_series - 1

def calculate_log_return_series(series: pd.Series) -> pd.Series:
    """
    Same as calculate_return_series but with log returns
    
    Args:
        series (pd.Series): Price series with datetime index
        
    Returns:
        pd.Series: Log return series where first value will be NaN
    """
    shifted_series = series.shift(1, axis=0)
    return pd.Series(np.log(series / shifted_series))

def calculate_percent_return(series: pd.Series) -> float:
    """
    Takes the first and last value in a series to determine the percent return,
    assuming the series is in date-ascending order
    
    Args:
        series (pd.Series): Price series with datetime index
        
    Returns:
        float: Total percentage return
    """
    return series.iloc[-1] / series.iloc[0] - 1

def get_years_past(series: pd.Series) -> float:
    """
    Calculate the years past according to the index of the series for use with
    functions that require annualization
    
    Args:
        series (pd.Series): Time series with datetime index
        
    Returns:
        float: Number of years between first and last date
    """
    start_date = series.index[0]
    end_date = series.index[-1]
    return (end_date - start_date).days / 365.25

def calculate_cagr(series: pd.Series) -> float:
    """
    Calculate compounded annual growth rate
    
    Args:
        series (pd.Series): Price series with datetime index
        
    Returns:
        float: Compound annual growth rate
    """
    start_price = series.iloc[0]
    end_price = series.iloc[-1]
    value_factor = end_price / start_price
    year_past = get_years_past(series)
    return (value_factor ** (1 / year_past)) - 1

def calculate_annualized_volatility(return_series: pd.Series) -> float:
    """
    Calculates annualized volatility for a date-indexed return series.
    Works for any interval of date-indexed prices and returns.
    
    Args:
        return_series (pd.Series): Return series with datetime index
        
    Returns:
        float: Annualized volatility
    """
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    return return_series.std() * np.sqrt(entries_per_year)

def calculate_sharpe_ratio(price_series: pd.Series, benchmark_rate: float = 0) -> float:
    """
    Calculates the Sharpe ratio given a price series.
    
    Args:
        price_series (pd.Series): Price series with datetime index
        benchmark_rate (float): Risk-free rate, defaults to 0
        
    Returns:
        float: Sharpe ratio
    """
    # Calculate returns and remove NA values
    returns = calculate_return_series(price_series).dropna()
    
    # Convert annual benchmark rate to match return frequency
    n_periods = len(returns)
    years_past = get_years_past(returns)
    periods_per_year = n_periods / years_past
    period_benchmark_rate = (1 + benchmark_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - period_benchmark_rate
    
    # Calculate annualized mean and std of excess returns
    mean_excess_return = excess_returns.mean() * periods_per_year
    volatility = calculate_annualized_volatility(returns)
    
    if volatility == 0:
        return np.nan
        
    return mean_excess_return / volatility

def calculate_sortino_ratio(price_series: pd.Series, benchmark_rate: float = 0) -> float:
    """
    Calculates the Sortino ratio.
    
    Args:
        price_series (pd.Series): Price series with datetime index
        benchmark_rate (float): Risk-free rate, defaults to 0
        
    Returns:
        float: Sortino ratio
    """
    # Calculate returns and remove NA values
    returns = calculate_return_series(price_series).dropna()
    
    # Convert annual benchmark rate to match return frequency
    n_periods = len(returns)
    years_past = get_years_past(returns)
    periods_per_year = n_periods / years_past
    period_benchmark_rate = (1 + benchmark_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - period_benchmark_rate
    
    # Calculate annualized mean excess return
    mean_excess_return = excess_returns.mean() * periods_per_year
    
    # Calculate downside deviation
    downside_deviation = calculate_annualized_downside_deviation(returns, benchmark_rate)
    
    if downside_deviation == 0:
        return np.nan
        
    return mean_excess_return / downside_deviation

def calculate_annualized_downside_deviation(return_series: pd.Series, benchmark_rate: float = 0) -> float:
    """
    Calculates the downside deviation for use in the Sortino ratio.
    
    Args:
        return_series (pd.Series): Return series with datetime index
        benchmark_rate (float): Risk-free rate, defaults to 0
        
    Returns:
        float: Annualized downside deviation
    """
    years_past = get_years_past(return_series)
    entries_per_year = return_series.shape[0] / years_past
    adjusted_benchmark_rate = ((1 + benchmark_rate) ** (1 / entries_per_year)) - 1
    downside_series = adjusted_benchmark_rate - return_series
    downside_sum_of_squares = (downside_series[downside_series > 0] ** 2).sum()
    denominator = return_series.shape[0] - 1
    downside_deviation = np.sqrt(downside_sum_of_squares / denominator)
    return downside_deviation * np.sqrt(entries_per_year)

def calculate_pure_profit_score(price_series: pd.Series) -> float:
    """
    Calculates the pure profit score
    
    Args:
        price_series (pd.Series): Price series with datetime index
        
    Returns:
        float: Pure profit score (CAGR * R²)
    """
    cagr = calculate_cagr(price_series)
    t: np.ndarray = np.arange(0, price_series.shape[0]).reshape(-1, 1)
    regression = LinearRegression().fit(t, price_series)
    r_squared = regression.score(t, price_series)
    return cagr * r_squared

def calculate_jensens_alpha(return_series: pd.Series, benchmark_return_series: pd.Series) -> float:
    """
    Calculates Jensen's alpha. Prefers input series have the same index. Handles NAs.
    
    Args:
        return_series (pd.Series): Return series with datetime index
        benchmark_return_series (pd.Series): Benchmark return series with datetime index
        
    Returns:
        float: Jensen's alpha
    """
    df = pd.concat([return_series, benchmark_return_series], sort=True, axis=1)
    df = df.dropna()
    clean_returns: pd.Series = df[df.columns.values[0]]
    clean_benchmarks = pd.DataFrame(df[df.columns.values[1]])
    regression = LinearRegression().fit(clean_benchmarks, y=clean_returns)
    return regression.intercept_

def calculate_jensens_alpha_v2(return_series: pd.Series) -> float:
    """
    Calculates Jensen's alpha using SPY as benchmark.
    
    Args:
        return_series (pd.Series): Return series with datetime index
        
    Returns:
        float: Jensen's alpha
    """
    start_date = return_series.index[0].strftime('%Y-%m-%d')
    end_date = return_series.index[-1].strftime('%Y-%m-%d')
    spy = yf.download('SPY', start=start_date, end=end_date)
    benchmark_return_series = calculate_log_return_series(spy['Close'])
    return calculate_jensens_alpha(return_series, benchmark_return_series)

DRAWDOWN_EVALUATORS: Dict[str, Callable] = {
    'dollar': lambda price, peak: peak - price,
    'percent': lambda price, peak: -((price / peak) - 1),
    'log': lambda price, peak: np.log(peak) - np.log(price),
}

def calculate_drawdown_series(series: pd.Series, method: str = 'log') -> pd.Series:
    """
    Returns the drawdown series
    
    Args:
        series (pd.Series): Price series with datetime index
        method (str): One of ['dollar', 'percent', 'log'], defaults to 'log'
        
    Returns:
        pd.Series: Drawdown series
    """
    if method not in DRAWDOWN_EVALUATORS:
        raise ValueError(f"Method must be one of {list(DRAWDOWN_EVALUATORS.keys())}")
    
    evaluator = DRAWDOWN_EVALUATORS[method]
    drawdown_values = []
    running_peak = series.iloc[0]
    
    for price in series.values:
        if price > running_peak:
            running_peak = price
        drawdown = evaluator(price, running_peak)
        drawdown_values.append(drawdown)
    
    return pd.Series(drawdown_values, index=series.index)

def calculate_max_drawdown(series: pd.Series, method: str = 'log') -> float:
    """
    Simply returns the max drawdown as a float
    
    Args:
        series (pd.Series): Price series with datetime index
        method (str): One of ['dollar', 'percent', 'log'], defaults to 'log'
        
    Returns:
        float: Maximum drawdown
    """
    return calculate_drawdown_series(series, method).max()

def calculate_max_drawdown_with_metadata(series: pd.Series, method: str = 'log') -> Dict[str, Any]:
    """
    Calculates max_drawndown and stores metadata about when and where.
    
    Args:
        series (pd.Series): Price series with datetime index
        method (str): One of ['dollar', 'percent', 'log'], defaults to 'log'
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - max_drawdown: float
            - peak_date: pd.Timestamp
            - peak_price: float
            - trough_date: pd.Timestamp
            - trough_price: float
    """
    if method not in DRAWDOWN_EVALUATORS:
        raise ValueError(f"Method must be one of {list(DRAWDOWN_EVALUATORS.keys())}")
    
    evaluator = DRAWDOWN_EVALUATORS[method]
    max_drawdown = 0
    peak_date = series.index[0]
    trough_date = series.index[0]
    recovery_date = None
    local_peak = series.iloc[0]
    peak_price = series.iloc[0]
    trough_price = series.iloc[0]
    
    for date, price in zip(series.index, series.values):
        if price > local_peak:
            local_peak = price
            if max_drawdown == 0:
                peak_date = date
                peak_price = price
        
        drawdown = evaluator(price, local_peak)
        
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            peak_date = date
            peak_price = local_peak
            trough_date = date
            trough_price = price
            recovery_date = None
        elif max_drawdown > 0 and price >= peak_price and recovery_date is None:
            recovery_date = date
    
    return {
        'max_drawdown': max_drawdown,
        'peak_date': peak_date,
        'peak_price': peak_price,
        'trough_date': trough_date,
        'trough_price': trough_price,
        'recovery_date': recovery_date
    }

def calculate_log_max_drawdown_ratio(series: pd.Series) -> float:
    """
    Calculate the ratio between log returns and log max drawdown
    
    Args:
        series (pd.Series): Price series with datetime index
        
    Returns:
        float: Log max drawdown ratio
    """
    log_drawdown = calculate_max_drawdown(series, method='log')
    log_return = np.log(series.iloc[-1]) - np.log(series.iloc[0])
    return log_return - log_drawdown

def calculate_calmar_ratio(series: pd.Series, years_past: int = 3) -> float:
    """
    Return the percent max drawdown ratio over the past three years (Calmar Ratio)
    
    Args:
        series (pd.Series): Price series with datetime index
        years_past (int): Number of years to look back, defaults to 3
        
    Returns:
        float: Calmar ratio
    """
    last_date = series.index[-1]
    three_years_ago = last_date - pd.Timedelta(days=years_past * 365.25)
    series = series[series.index > three_years_ago]
    percent_drawdown = calculate_max_drawdown(series, method='percent')
    cagr = calculate_cagr(series)
    return cagr / percent_drawdown

def calculate_rolling_sharpe_ratio(price_series: pd.Series, n: float = 20) -> pd.Series:
    """
    Compute an approximation of the Sharpe ratio on a rolling basis.
    Intended for use as a preference value.
    
    Args:
        price_series (pd.Series): Price series with datetime index
        n (float): Rolling window size, defaults to 20
        
    Returns:
        pd.Series: Rolling Sharpe ratio
    """
    rolling_return_series = calculate_return_series(price_series).rolling(n)
    return rolling_return_series.mean() / rolling_return_series.std()
