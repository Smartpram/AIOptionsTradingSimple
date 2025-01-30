import logging
from logging_config import configure_logging
import talib
import pandas as pd

logger = configure_logging()

def technical_analysis(stock_data: pd.DataFrame) -> pd.DataFrame:
    """Performs technical analysis and generates trading signals.

    Args:
        stock_data (DataFrame): Historical stock data.

    Returns:
        DataFrame: Stock data with technical indicators and buy/sell signals.
    """
    try:
        # Compute technical indicators
        stock_data['EVWMA'] = (stock_data['Volume'] * stock_data['Close']).cumsum() / stock_data['Volume'].cumsum()
        stock_data['MACD'], stock_data['MACD_Signal'], _ = talib.MACD(stock_data['Close'], 
                                                                      fastperiod=12, slowperiod=26, signalperiod=9)
        stock_data['VWAP'] = (stock_data['Volume'] * stock_data['Close']).cumsum() / stock_data['Volume'].cumsum()

        stock_data['Stochastic_RSI'] = ((stock_data['Close'] - stock_data['Close'].rolling(14).min()) /
                                         (stock_data['Close'].rolling(14).max() - stock_data['Close'].rolling(14).min())).rolling(14).mean()

        # Generate signals
        stock_data['Signal'] = 0
        stock_data.loc[(stock_data['MACD'] > stock_data['MACD_Signal']) & 
                       (stock_data['Close'] > stock_data['EVWMA']) & 
                       (stock_data['Stochastic_RSI'] < 0.2), 'Signal'] = 1  # Buy signal
                       
        stock_data.loc[(stock_data['MACD'] < stock_data['MACD_Signal']) & 
                       (stock_data['Close'] < stock_data['EVWMA']) & 
                       (stock_data['Stochastic_RSI'] > 0.8), 'Signal'] = -1  # Sell signal

        logger.info("Technical analysis completed.")
        return stock_data
    except Exception as e:
        logger.error(f"Error performing technical analysis: {e}")
        return None
