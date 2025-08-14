from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=1, type=int, metavar='N',
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
    
    # # Convert negative values to 0 and apply log(x + 1) transformation
    # df[df < 0] = 0
    # df = df.applymap(lambda x: np.log(x + 1))
    
    return df
# Define pipeline 2 processing function for a single column in columns_pipeline_2
def process_column_in_pipeline_2(column):
    # Read the specific column from the Parquet file
    df = pd.read_parquet(parquet_file, columns=[column])
    
    # # Convert from HEX to int and apply positive modulus over 5000
    # df = df.applymap(lambda x: int(x, 16) % 5000 if isinstance(x, str) else x % 5000)
    
    return df

# Define the main function
def main():
    args = parser.parse_args()
    # print("Number of jobs: ", args.n_jobs)

    start_time = time.time()  # Record the start time

    # Define the column sets for the two pipelines
    # columns_pipeline_0 = [f"col_{i}" for i in range(0, 1)]  # 1 column, target
    # columns_pipeline_1 = [f"col_{i}" for i in range(1, 14)]  # 13 columns, dense features
    columns_pipeline_2 = [f"col_{i}" for i in range(14, 15)]  # 26 columns, sparse features

    # Use joblib's Parallel to process each column separately, generating 39 tasks (one per column)
    # result = Parallel(n_jobs=args.n_jobs)(
    #     [delayed(process_column_in_pipeline_1)(col) for col in columns_pipeline_1] +  # 13 tasks for pipeline 1
    #     [delayed(process_column_in_pipeline_2)(col) for col in columns_pipeline_2]    # 26 tasks for pipeline 2
    # )
    result = Parallel(n_jobs=args.n_jobs)(
        # [delayed(process_column_in_pipeline_0)(col) for col in columns_pipeline_0]
        # [delayed(process_column_in_pipeline_1)(col) for col in tqdm(columns_pipeline_1)] +  
        [delayed(process_column_in_pipeline_2)(col) for col in tqdm(columns_pipeline_2)]
    )

    # # Verify the shape of each processed column
    # for i, df in enumerate(result):
    #     print(f"Shape of DataFrame for column {i}: {df.shape}")

    # # Concatenate the results back into a single DataFrame for each pipeline
    # df_pipeline_1 = pd.concat(result[:13], axis=1)  # First 13 tasks are for pipeline 1
    # df_pipeline_2 = pd.concat(result[13:], axis=1)  # Last 26 tasks are for pipeline 2

    # print("Columns in pipeline 1 DataFrame:", df_pipeline_1.columns)
    # print("Columns in pipeline 2 DataFrame:", df_pipeline_2.columns)

    # # Print the first row from both DataFrames
    # print("First row from the first pipeline (after processing):")
    # print(df_pipeline_1.iloc[0])

    # print("\nFirst row from the second pipeline (after processing):")
    # print(df_pipeline_2.iloc[0])
    # # print(df_pipeline_2.iloc[0].apply(lambda x: hex(x & 0xFFFFFFFF)))

    # # Count the number of rows for each pipeline
    # num_rows_pipeline_1 = len(df_pipeline_1)
    # num_rows_pipeline_2 = len(df_pipeline_2)

    # print(f"\nNumber of rows in the first pipeline: {num_rows_pipeline_1}")
    # print(f"Number of rows in the second pipeline: {num_rows_pipeline_2}")

    # Record the end time and calculate the total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

# Entry point for the program
if __name__ == "__main__":
    main()
