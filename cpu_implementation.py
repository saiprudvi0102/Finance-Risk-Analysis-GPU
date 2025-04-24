import numpy as np
import pandas as pd
import time

def calculate_daily_returns(stock_data):
    """Calculate the daily returns for each stock."""
    return stock_data.pct_change().dropna()

def calculate_var_cpu(returns, confidence_level=0.05):
    """Calculate Value at Risk using historical method."""
    return np.percentile(returns, confidence_level * 100)

def calculate_covariance_matrix_cpu(stock_returns):
    """Calculate the covariance matrix using CPU."""
    start_time = time.time()
    
    # Initialize an empty DataFrame for the covariance matrix
    stocks = stock_returns.columns
    covariance_matrix = pd.DataFrame(index=stocks, columns=stocks)
    
    # Fill the diagonal with variances
    for stock in stocks:
        covariance_matrix.at[stock, stock] = np.var(stock_returns[stock], ddof=0)
    
    # Fill the off-diagonal elements with covariances
    for i, stock1 in enumerate(stocks):
        for stock2 in stocks[i+1:]:
            returns1 = stock_returns[stock1]
            returns2 = stock_returns[stock2]
            cov = np.cov(returns1, returns2, ddof=0)[0][1]
            covariance_matrix.at[stock1, stock2] = cov
            covariance_matrix.at[stock2, stock1] = cov
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return covariance_matrix, execution_time

def run_cpu_analysis(file_path):
    """Run the complete CPU-based analysis."""
    total_start_time = time.time()
    
    # Load stock price data
    stock_data = pd.read_csv(file_path)
    
    # Calculate daily returns
    stock_returns = calculate_daily_returns(stock_data)
    
    # Calculate VaR for each stock
    var_results = {}
    for stock in stock_returns.columns:
        var_results[stock] = calculate_var_cpu(stock_returns[stock])
    
    # Calculate covariance matrix
    covariance_matrix, cov_time = calculate_covariance_matrix_cpu(stock_returns)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    results = {
        "covariance_time": cov_time,
        "total_execution_time": total_execution_time,
        "var_results": var_results,
        "covariance_matrix": covariance_matrix
    }
    
    return results
