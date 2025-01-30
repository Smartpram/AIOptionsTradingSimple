import logging
from logging_config import configure_logging
import pandas as pd
from options_trading.data_acquisition import acquire_data
from options_trading.technical_analysis import technical_analysis
from options_trading.quantitative_analysis import feature_engineering
from options_trading.model import create_model
from options_trading.backtest import backtest_strategy

logger = configure_logging()

def main():
    """Main function to execute the options trading strategy."""
    try:
        ticker = input("Enter the stock ticker (e.g., JPM): ")
        expiration_dates_input = input("Enter the expiration dates for options (comma separated, e.g., 2023-01-20,2023-02-17): ")
        expiration_dates = [pd.to_datetime(date.strip()) for date in expiration_dates_input.split(',')]

        stock_data, calls, puts = acquire_data(ticker, expiration_dates)
        
        if calls.empty and puts.empty:
            logger.warning("No options data available. Please check the expiration dates or ticker.")
            return

        stock_data = technical_analysis(stock_data)
        features_calls = feature_engineering(stock_data, calls)
        features_puts = feature_engineering(stock_data, puts)

        all_features = pd.concat([features_calls, features_puts]).reset_index(drop=True)
        
        if all_features.empty or stock_data.empty:
            logger.warning("No feature data available for modeling.")
            return

        # Scale Features
        scaler = StandardScaler()
        X = scaler.fit_transform(all_features.drop(columns=['IV_Percentile']).values)
        
        # Add dummy target variable for demonstration purposes
        y = np.random.rand(len(X)) * 100
        all_features['predicted_price'] = y

        # Train/Test Split
        split_index = int(0.8 * len(X))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Create and train model
        model = create_model((X_train.shape[1],))
        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)

        # Predictions
        predictions = model.predict(X_test)
        all_features.loc[split_index:, 'predicted_price'] = predictions.flatten()

        # Backtest Strategy
        final_portfolio_value = backtest_strategy(stock_data)
        logger.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
