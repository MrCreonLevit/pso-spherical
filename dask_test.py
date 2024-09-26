import os
import time
import sys
from datetime import datetime
from time import sleep
import numpy as np

from numba import jit, njit
import dask

@dask.delayed
def work(i):
    sleep(5)
    print(f"Finished work {i}", flush=True)
    return i*i


# Set up Dask client
# client = Client()

# Number of PSO runs
num_runs = 10


# Optimize test functions
if __name__ == "__main__":
    # Set a fixed seed for reproducibility. You can use any integer value as the seed
    np.random.seed(int(time.time()))
    print("Optimizing minimum angle with PSO:",flush=True)
    results = []
    for i in range(num_runs):
        #print(f"Queing up PSO {i+1} of {num_runs}",flush=True)
        #y = dask.delayed(pso)(min_angle, 4, 12, population_size=30, max_iterations=100, c2=0.0, plots=False)
        y = dask.delayed(work)(i)
        results.append(y)
    results = dask.compute(*results)    
    print(results)    