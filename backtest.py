def backtest_strategy(stock_data):
    """Simple backtesting of the strategy."""
    initial_capital = 10000
    cash, shares = initial_capital, 0
    
    for index, row in stock_data.iterrows():
        signal = row['Signal']
        predicted_price = row['Close']

        if signal == 1 and cash > 0:  # Buy signal
            shares += 1
            cash -= predicted_price
        elif signal == -1 and shares > 0:  # Sell signal
            cash += predicted_price
            shares -= 1
    
    total_portfolio_value = cash + shares * stock_data['Close'].iloc[-1]
    return total_portfolio_value
