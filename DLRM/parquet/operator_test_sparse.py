import pandas as pd
import numpy as np
import time
from joblib import Parallel, delayed
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=8, type=int, metavar='N',
                    help='number of total jobs to run')
parser.add_argument('--modulus', default=8192, type=int, metavar='M',
                    help='modulus value for sparse features')

# Path to your Parquet file
parquet_file = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"

# Function to convert HEX to integers
def hex_to_int(df):
    start_time = time.time()
    # Convert from HEX to integers
    df = df.applymap(lambda x: int(x, 16) if isinstance(x, str) else x)
    elapsed_time = time.time() - start_time
    print(f"Time for HEX to int conversion: {elapsed_time:.4f} seconds")
    return df

# Function to apply modulus
def apply_modulus(df, modulus):
    start_time = time.time()
    # Apply modulus operation
    df = df.applymap(lambda x: x % modulus if isinstance(x, int) else x)
    elapsed_time = time.time() - start_time
    print(f"Time for modulus operation: {elapsed_time:.4f} seconds")
    return df

# Function to create mapping from unique values and apply it for multiple columns
def apply_mapping(df, columns):
    
    vocab_gen_time = 0
    vocab_apply_time = 0
    # Extract unique values and create a mapping table for each column
    for column in columns:

        start_time = time.time()
        unique_values = df[column].unique()
        unique_value_to_index = {val: idx for idx, val in enumerate(unique_values)}
        unique_time = time.time()

        df[column] = df[column].map(unique_value_to_index)
        end_time = time.time()

        vocab_gen_time += unique_time - start_time
        vocab_apply_time += end_time - unique_time

    print(f"Time for generating vocab: {vocab_gen_time:.4f} seconds")
    print(f"Time for applying vocab: {vocab_apply_time:.4f} seconds")
    return df


# Function to read the specified column from the Parquet file
def read_column_from_parquet(columns):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=columns)
    return df

# Main function to benchmark each operator independently
def main():
    args = parser.parse_args()
    modulus = args.modulus

    # Define the column to be used for testing
    column_to_test = [f'col_{x}' for x in range(14, 40)]  # Example column name

    # Read the data for the specified column
    df = read_column_from_parquet(column_to_test)

    # Benchmark the HEX to int conversion
    hex_int_df = hex_to_int(df.copy())

    # Benchmark the modulus operation
    modulus_df = apply_modulus(hex_int_df.copy(), modulus)

    # Benchmark the mapping operation
    mapping_df = apply_mapping(modulus_df.copy(), column_to_test)

    # Print final dataset after mapping (first 5 rows)
    print(f"Final dataset after mapping (first 5 rows): \n{mapping_df.head()}")

# Entry point for the program
if __name__ == "__main__":
    main()
