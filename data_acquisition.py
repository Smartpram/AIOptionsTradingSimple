import yfinance as yf
import pandas as pd

def acquire_data(ticker, expiration_dates):
    """Fetch historical stock prices and option data."""
    stock_data = yf.download(ticker, start='2018-01-01', end='2023-01-01')
    calls, puts = [], []

    for date in expiration_dates:
        try:
            option_chain = yf.Ticker(ticker).option_chain(date)
            calls.append(option_chain.calls)
            puts.append(option_chain.puts)
        except Exception as e:
            print(f"Error fetching options data for {date}: {e}")

    return stock_data, pd.concat(calls) if calls else pd.DataFrame(), pd.concat(puts) if puts else pd.DataFrame()
