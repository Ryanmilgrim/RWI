"""
Created on Fri Dec 15 20:52:27 2023
@author: ryanm
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from datetime import date, timedelta

# %% Data Collection

weekly_macro = {
    'WM2NS': {  # M2 Money Supply
        'pct_change': {'periods': 26}  # 4-week percent change
    },
    'NFCI': {  # Chicago Financial Conditions Index
        'clip': {'upper': 1.0}  # No transformation, use as is
    },
    'CCSA': {  # Continued Claims
        'clip': {'upper': 8000000}  # No transformation, use as is
    },
    'DPSACBW027SBOG': {  # Deposits, All Commercial Banks
        'pct_change': {'periods': 26}  # 4-week percent change
    },
    'TOTBKCR': {  # Bank Credit, All Commercial Banks
        'pct_change': {'periods': 26}  # 4-week percent change
    },
    'BOPGSTB': {  # Trade Balance
        'diff': {'periods': 52}  # 12-week percent change
    },
    'PCE': { # Personal Consumption
        'pct_change': {'periods': 26}, 'clip': {'lower': -0.05, 'upper': 0.1}  # 12-week percent change
    }
}


def getReturns(assets=None, years=15, freq='B', end_date=None):
    """
    Fetches and calculates stock returns for a specified number of years back from a given end date,
    resampled according to a specified frequency, using ticker symbols from a CSV file.

    Parameters:
    assets (list): List of ETFs to provide the yfinance api. If None a list of default assets is used.
    years (float): Number of years back from the end date for data.
    freq (str): 'B' for business days or 'M' for monthly returns. Defaults to 'B'.
    end_date (str, optional): End date for data, defaults to today's date if not provided.

    Returns:
    DataFrame of resampled returns
    """

    if assets is None:
        assets = ['SPY', 'IJH', 'IJR', 'LQD', 'HYG', 'SPTL', 'GLD']

    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=365.25 * years)

    prices = yf.download(assets, start=start_date, end=end_date)['Adj Close']
    returns = prices.interpolate().resample(freq).last().pct_change().to_period().dropna()
    return returns[returns.index.end_time.date < date.today()][assets]


def getMacro(macro=weekly_macro, freq='M', years=40, end_date=None):
    """
    Fetches data from FRED and applies specified transformations based on given parameters.

    Parameters:
    tickers (list): List of FRED ticker symbols.
    macro (dict): Dictionary where keys are FRED tickers and values are dictionaries specifying transformations and parameters.
    years (int): Number of years back from the end date for data. Defaults to 5.
    end_date (str, optional): End date for the data, defaults to the current date.

    Returns:
    pd.DataFrame: DataFrame with transformed FRED data.
    """
    
    tickers = list(macro.keys())
    if end_date is None:
        end_date = date.today()
    start_date = end_date - timedelta(days=365.25 * (years + 2))
    raw_data = pdr.DataReader(tickers, 'fred', start_date, end_date).resample(freq).last().ffill()
    raw_data.index = pd.PeriodIndex(raw_data.index, freq=freq)

    data = pd.DataFrame()
    for ticker in tickers:
        timeseries = raw_data[ticker]
        for method, params in macro[ticker].items():
            if None:
                continue
            try:
                timeseries = getattr(timeseries, method)
                timeseries = timeseries() if len(params) == 0 else timeseries(**params)
            except TypeError:
                timeseries = pd.Series(
                    method(timeseries), timeseries.index, dtype=float, name=timeseries.name)

        data[ticker] = timeseries
    return data.dropna()


# %% Calculations


def CalculateReturnStatistics(returns):
    """
    Calculate basic statistics of a given financial returns series.

    Parameters:
    - returns (pd.Series): A pandas Series representing the asset returns.

    Returns:
    - pd.Series: A pandas Series containing the following statistics:
        - Observations: The number of data points.
        - Mean: The average return.
        - Vol: The standard deviation (volatility) of the returns.
        - Skew: The skewness of the returns.
        - Kurtosis: The excess kurtosis of the returns (Kurtosis - 3).
    """
    return pd.Series({
        'Observations': len(returns),
        'Mean': returns.mean(),
        'Vol': returns.std(),
        'Skew': returns.skew(),
        'Kurtosis': returns.kurtosis() + 3  # Excess kurtosis
    })


def cov2cor(cov):
    """This does the thing"""
    vol = pd.Series(np.diag(cov) ** 0.5, cov.index)
    cor = pd.DataFrame(np.diag(1 / vol) @ cov @ np.diag( 1 / vol))
    cor.index, cor.columns = cov.index, cov.columns
    return vol, cor


