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
parquet_file = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"

# Function to generate a pseudo dataset
def generate_pseudo_dataset(num_rows, num_cols):
    # Random dense data with negative values
    data = np.random.uniform(-1000, 1000, size=(num_rows, num_cols)).astype(np.float32)
    df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(num_cols)])
    return df

# Define pipeline processing functions
def apply_negative_to_zero(df):
    # Apply df[df < 0] = 0
    start_time = time.time()
    df[df < 0] = 0
    elapsed_time = time.time() - start_time
    print(f"Time to apply df[df < 0] = 0: {elapsed_time:.4f} seconds")
    return df

def apply_log_transform(df):
    # Apply log(x + 1) transformation
    start_time = time.time()
    df = df.applymap(lambda x: np.log(x + 1))
    elapsed_time = time.time() - start_time
    print(f"Time to apply log(x + 1): {elapsed_time:.4f} seconds")
    return df

# Function to read the specified column from the Parquet file
def read_column_from_parquet(columns):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=columns)
    return df

# Define the main function
def main():
    args = parser.parse_args()
    modulus = args.modulus
    # num_rows = args.rows
    # num_cols = 1  # Adjust the number of columns as per your dataset
    
    # # Generate pseudo dataset
    # df = generate_pseudo_dataset(num_rows, num_cols)

    # Define the column to be used for testing
    column_to_test = [f'col_{x}' for x in range(1, 14)]  # Example column name

    # Read the data for the specified column
    df = read_column_from_parquet(column_to_test)

    # Benchmark df[df < 0] = 0
    df_zeroed = apply_negative_to_zero(df.copy())

    # Benchmark log transformation
    df_logged = apply_log_transform(df_zeroed.copy())

    # The final dataset with the transformations applied (df_logged)
    print(f"Final dataset shape: {df_logged.shape}")

# Entry point for the program
if __name__ == "__main__":
    main()
