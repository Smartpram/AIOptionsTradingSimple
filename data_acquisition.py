import logging
from logging_config import configure_logging
import pandas as pd
import yfinance as yf

logger = configure_logging()

def acquire_data(ticker: str, expiration_dates: list) -> tuple:
    """Fetches historical stock and options data for a given ticker.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        expiration_dates (list): List of expiration date strings.

    Returns:
        tuple: A tuple containing stock_data (DataFrame), calls (DataFrame), and puts (DataFrame).
    """
    try:
        stock_data = yf.download(ticker, start='2018-01-01', end='2023-01-01')
        calls, puts = [], []

        for date in expiration_dates:
            try:
                option_chain = yf.Ticker(ticker).option_chain(date)
                calls.append(option_chain.calls)
                puts.append(option_chain.puts)
            except Exception as e:
                logger.error(f"Error fetching options data for {date}: {e}")

        logger.info("Data acquisition completed.")
        return stock_data, pd.concat(calls) if calls else pd.DataFrame(), pd.concat(puts) if puts else pd.DataFrame()
    except Exception as e:
        logger.error(f"Error acquiring data: {e}")
        return None, None, None
