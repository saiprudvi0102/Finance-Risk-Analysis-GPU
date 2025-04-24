import numpy as np
import pandas as pd
import os

def generate_synthetic_stock_data(num_stocks=10, num_days=1000000, seed=42):
    """
    Generate synthetic stock price data for multiple stocks over a specified number of days.
    
    Parameters:
    num_stocks (int): Number of stocks to simulate
    num_days (int): Number of days to simulate
    seed (int): Random seed for reproducibility
    
    Returns:
    pd.DataFrame: DataFrame with simulated stock prices
    """
    np.random.seed(seed)
    
    # Initial stock prices
    initial_prices = np.random.uniform(10, 500, num_stocks)
    
    # Daily returns with slight correlation
    daily_returns = np.random.normal(0.0005, 0.02, (num_days, num_stocks))
    
    # Generate price paths
    stock_prices = np.zeros((num_days, num_stocks))
    stock_prices[0] = initial_prices
    
    for t in range(1, num_days):
        stock_prices[t] = stock_prices[t-1] * (1 + daily_returns[t])
    
    # Create DataFrame
    columns = [f"Stock{i+1}" for i in range(num_stocks)]
    df = pd.DataFrame(stock_prices, columns=columns)
    
    return df

def save_dataset(df, size, output_dir="./data"):
    """Save generated dataset to CSV file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepath = os.path.join(output_dir, f"stock_prices_{size}_days.csv")
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    return filepath

def main():
    # Generate datasets of different sizes
    sizes = [10000, 100000, 1000000]
    
    for size in sizes:
        print(f"Generating dataset with {size} days...")
        df = generate_synthetic_stock_data(num_days=size)
        save_dataset(df, size)

if __name__ == "__main__":
    main()
