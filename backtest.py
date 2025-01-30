import logging
from logging_config import configure_logging
import pandas as pd

logger = configure_logging()

def backtest_strategy(stock_data: pd.DataFrame) -> float:
    """Backtests the trading strategy based on buy/sell signals.

    Args:
        stock_data (DataFrame): DataFrame containing stock data and signals.

    Returns:
        float: Final portfolio value after backtesting.
    """
    try:
        initial_capital = 10000
        cash, shares = initial_capital, 0
        
        for _, row in stock_data.iterrows():
            signal = row['Signal']
            predicted_price = row['Close']

            if signal == 1 and cash > 0:  # Buy signal
                shares += 1
                cash -= predicted_price
            elif signal == -1 and shares > 0:  # Sell signal
                cash += predicted_price
                shares -= 1
        
        total_portfolio_value = cash + shares * stock_data['Close'].iloc[-1]
        logger.info(f"Backtesting completed. Final portfolio value: ${total_portfolio_value:.2f}")
        return total_portfolio_value
    except Exception as e:
        logger.error(f"Error backtesting strategy: {e}")
        return None
