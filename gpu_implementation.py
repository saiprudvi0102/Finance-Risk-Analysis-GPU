import numpy as np
import pandas as pd
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# CUDA kernel for covariance calculation
kernel_code = """
__global__ void covariance_kernel(float *returns1, float *returns2, float *result, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        atomicAdd(result, (returns1[idx] - returns1[N]) * (returns2[idx] - returns2[N]));
    }
}
"""

def calculate_daily_returns(stock_data):
    """Calculate the daily returns for each stock."""
    return stock_data.pct_change().dropna()

def calculate_var_gpu(returns, confidence_level=0.05):
    """Calculate Value at Risk using historical method."""
    return np.percentile(returns, confidence_level * 100)

def calculate_covariance_gpu(returns1, returns2):
    """Calculate covariance between two return series using GPU."""
    returns1 = np.array(returns1, dtype=np.float32)
    returns2 = np.array(returns2, dtype=np.float32)
    N = len(returns1)
    result = np.zeros(1, dtype=np.float32)
    
    # Calculate means
    returns1_mean = np.mean(returns1)
    returns2_mean = np.mean(returns2)
    
    # Append means to the end of the arrays
    returns1 = np.append(returns1, returns1_mean)
    returns2 = np.append(returns2, returns2_mean)
    
    # Allocate memory on the GPU
    returns1_gpu = cuda.mem_alloc(returns1.nbytes)
    returns2_gpu = cuda.mem_alloc(returns2.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)
    
    # Transfer data to the GPU
    cuda.memcpy_htod(returns1_gpu, returns1)
    cuda.memcpy_htod(returns2_gpu, returns2)
    cuda.memcpy_htod(result_gpu, result)
    
    # Compile and get the kernel function
    module = SourceModule(kernel_code)
    covariance_kernel = module.get_function("covariance_kernel")
    
    # Calculate grid and block dimensions
    block_size = 256
    grid_size = min(200, (N + block_size - 1) // block_size)
    
    # Launch kernel
    covariance_kernel(
        returns1_gpu, returns2_gpu, result_gpu, np.int32(N),
        block=(block_size, 1, 1), 
        grid=(grid_size, 1),
        shared=(block_size * 4)  # 4 bytes per float
    )
    
    # Copy the result back to the host
    cuda.memcpy_dtoh(result, result_gpu)
    
    # Free GPU memory
    returns1_gpu.free()
    returns2_gpu.free()
    result_gpu.free()
    
    # The kernel calculates the sum, so to get the covariance we divide by N-1
    return result[0] / (N - 1)

def calculate_covariance_matrix_gpu(stock_returns):
    """Calculate the covariance matrix using GPU."""
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
            cov = calculate_covariance_gpu(returns1, returns2)
            covariance_matrix.at[stock1, stock2] = cov
            covariance_matrix.at[stock2, stock1] = cov
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return covariance_matrix, execution_time

def run_gpu_analysis(file_path):
    """Run the complete GPU-based analysis."""
    total_start_time = time.time()
    
    # Load stock price data
    stock_data = pd.read_csv(file_path)
    
    # Calculate daily returns
    stock_returns = calculate_daily_returns(stock_data)
    
    # Calculate VaR for each stock
    var_results = {}
    for stock in stock_returns.columns:
        var_results[stock] = calculate_var_gpu(stock_returns[stock])
    
    # Calculate covariance matrix
    covariance_matrix, cov_time = calculate_covariance_matrix_gpu(stock_returns)
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    results = {
        "covariance_time": cov_time,
        "total_execution_time": total_execution_time,
        "var_results": var_results,
        "covariance_matrix": covariance_matrix
    }
    
    return results
