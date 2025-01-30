import numpy as np
import pandas as pd
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    """Calculate call and put option prices using the Black-Scholes formula."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return call_price, put_price

def calculate_greeks(S, K, T, r, sigma):
    """Calculate option greeks."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, theta, vega, rho

def feature_engineering(stock_data, option_data):
    """Feature engineering for model input."""
    option_data['Moneyness'] = option_data['strike'] / stock_data['Close'].iloc[-1]
    option_data['Days_to_Expiry'] = (option_data['expiration'] - pd.to_datetime('now')).dt.days / 365.0
    option_data['Implied_Volatility'] = option_data['impliedVolatility'].fillna(0)

    historical_iv = option_data['Implied_Volatility'].describe(percentiles=[0.00, 0.05, 0.20, 0.35, 0.65, 0.80, 0.95, 1.00])
    
    option_data['IV_Percentile'] = pd.cut(
        option_data['Implied_Volatility'], 
        bins=[-np.inf] + historical_iv['25%'].tolist() + [np.inf],
        labels=['Cheap', 'Cheap', 'Cheap', 'Cheap', 'Expensive', 'Expensive', 'Expensive', 'Expensive']
    )

    # Calculate Greeks
    stock_price = stock_data['Close'].iloc[-1]
    r = 0.01
    greeks_data = [calculate_greeks(stock_price, row['strike'], row['Days_to_Expiry'], r, row['Implied_Volatility']) for index, row in option_data.iterrows()]
    
    option_data[['delta', 'gamma', 'theta', 'vega', 'rho']] = pd.DataFrame(greeks_data)

    return option_data[['strike', 'Moneyness', 'Days_to_Expiry', 'Implied_Volatility', 'IV_Percentile', 'delta', 'gamma', 'theta', 'vega', 'rho']]
