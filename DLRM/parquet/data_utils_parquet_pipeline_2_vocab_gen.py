from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=8, type=int, metavar='N',
                    help='number of total jobs to run')
parser.add_argument('--modulus', default=8192, type=int, metavar='M',
                    help='modulus value for sparse features')

# Path to your Parquet file
parquet_file = "/local/home/yuzhuyu/bin2parquet.parquet"

# Define pipeline 0 processing function for a single column in columns_pipeline_0
def process_column_in_pipeline_0(column):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=[column])
    return df

# Define pipeline 1 processing function for a single column in columns_pipeline_1
def process_column_in_pipeline_1(column):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=[column])
    
    # Convert negative values to 0 and apply log(x + 1) transformation
    df[df < 0] = 0
    df = df.applymap(lambda x: np.log(x + 1))
    
    return df

# Define pipeline 2 processing function for a single column in columns_pipeline_2
def process_column_in_pipeline_2(column, modulus):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=[column])
    
    # Convert from HEX to int and apply positive modulus
    df = df.applymap(lambda x: int(x, 16) % modulus if isinstance(x, str) else x % modulus)

    # Extract unique values and create a mapping table
    unique_values = df[column].unique()
    
    # Create a mapping from unique values to their corresponding index
    unique_value_to_index = {val: idx for idx, val in enumerate(unique_values)}
    
    # # Map values to indices
    # df[column] = df[column].map(unique_value_to_index)
    
    return df

# Define the main function
def main():
    args = parser.parse_args()
    modulus = args.modulus

    start_time = time.time()

    # Define the column sets for the pipelines
    # columns_pipeline_0 = [f"col_{i}" for i in range(0, 1)]  # 1 column, target
    # columns_pipeline_1 = [f"col_{i}" for i in range(1, 14)]  # 13 columns, dense features
    columns_pipeline_2 = [f"col_{i}" for i in range(14, 15)]  # 26 columns, sparse features

    result = Parallel(n_jobs=args.n_jobs)(
        # [delayed(process_column_in_pipeline_0)(col) for col in columns_pipeline_0] +
        # [delayed(process_column_in_pipeline_1)(col) for col in columns_pipeline_1] +  
        [delayed(process_column_in_pipeline_2)(col, modulus) for col in columns_pipeline_2]
    )

    # Record the end time and calculate the total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

# Entry point for the program
if __name__ == "__main__":
    main()
