# GPU-Accelerated Financial Risk Analysis


## Project Overview

This project demonstrates the application of GPU computing to accelerate financial risk analysis, particularly focusing on the computation of covariance matrices for large financial datasets. The implementation compares traditional CPU-based methods with CUDA-powered GPU implementations to quantify performance improvements.

## Key Features

- **GPU Acceleration**: Leverages NVIDIA CUDA for parallel processing of financial calculations
- **Comparative Analysis**: Benchmarks CPU vs GPU performance across multiple dataset sizes
- **Financial Risk Metrics**: Calculates Value at Risk (VaR) and covariance matrices for portfolio risk assessment
- **Visualizations**: Provides charts and graphs to illustrate performance gains and financial insights
- **Scalability Testing**: Demonstrates how GPU advantages increase with larger datasets

## Background

In financial analytics, the covariance matrix is essential for portfolio optimization, risk assessment, and other statistical analyses. Traditional CPU computations can be time-intensive with large financial datasets. This project explores how GPUs can significantly improve computational efficiency.

## Results

Our benchmarks show:

| Dataset Size (Days) | CPU Time (s) | GPU Time (s) | Speedup Factor |
|---------------------|--------------|--------------|----------------|
| 10,000              | 0.06         | 0.07         | 0.86x          |
| 100,000             | 0.45         | 0.40         | 1.13x          |
| 1,000,000           | 5.57         | 3.55         | 1.57x          |

For the covariance matrix calculation specifically:
- With 10,000 days: CPU outperforms GPU (0.01s vs 0.03s)
- With 100,000 days: CPU still slightly faster (0.05s vs 0.07s)
- With 1,000,000 days: GPU shows significant advantage (0.43s vs 0.75s)

These results indicate that GPU acceleration becomes increasingly beneficial as dataset size grows, with the most substantial performance improvements observed with the largest datasets.

## Installation

### Prerequisites

- Python 3.8+
- CUDA Toolkit 11.0+
- NVIDIA GPU with CUDA support

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/gpu-financial-risk-analysis.git
   cd gpu-financial-risk-analysis
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Generate synthetic datasets (or use your own):
   ```
   python src/data_generator.py
   ```

## Usage

### Basic Usage

Run the performance comparison:
```
python src/benchmark.py
```

### Jupyter Notebooks

Explore the analysis in interactive notebooks:
```
jupyter notebook notebooks/performance_analysis.ipynb
```

## Project Structure

- `data/`: Contains stock price datasets of various sizes
- `src/`: Source code for CPU and GPU implementations
- `notebooks/`: Jupyter notebooks for analysis and visualization
- `results/`: Output visualizations and benchmark results

  ├── data/
│   └── stock_price_simulation.csv
├── src/
│   ├── cpu_implementation.py
│   ├── gpu_implementation.py
│   ├── data_generator.py
│   ├── visualization.py
│   └── utils.py
├── notebooks/
│   ├── performance_analysis.ipynb
│   └── financial_risk_metrics.ipynb
├── results/
│   ├── benchmarks/
│   │   └── performance_comparison.csv
│   └── visualizations/
│       └── cpu_vs_gpu_chart.png
├── requirements.txt
├── README.md
└── LICENSE
Implementation Details
CPU Implementation (cpu_implementation.py)

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
    
GPU Implementation (gpu_implementation.py)




## Future Directions

- Integration with machine learning algorithms for predictive analytics
- Exploration of cloud-based GPU computing solutions
- Extension to additional financial instruments and risk metrics
- Optimization of memory management for even larger datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by the research paper "Enhancing Financial Risk Analysis" by Ela and Juvvala
- Thanks to the CUDA and PyCUDA development teams
