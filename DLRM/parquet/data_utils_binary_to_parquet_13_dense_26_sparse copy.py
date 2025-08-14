import numpy as np
import pandas as pd
import random

def generate_pseudo_parquet(output_parquet, num_rows=45840617, target_column='col_0', num_dense=13, num_sparse=26):
    # Step 1: Generate the target column with binary values (0 or 1)
    target_data = np.random.randint(0, 2, size=num_rows)

    # Step 2: Generate dense features with random positive and negative float values as float32
    dense_data = np.random.uniform(-1000, 1000, size=(num_rows, num_dense)).astype(np.float32)

    # Step 3: Generate sparse features with random 32-bit integers and convert to hexadecimal
    sparse_data = np.array([[f"{random.randint(0, 0xFFFFFFFF):08X}" for _ in range(num_sparse)] for _ in range(num_rows)])

    # # Generate random 32-bit integers
    # sparse_data_int = np.random.randint(0, 2**32, size=(num_rows, num_sparse), dtype=np.uint32)
    
    # # Convert integers to hex strings with leading zeros and uppercase
    # sparse_data_hex = np.vectorize(lambda x: f"{x:08X}")(sparse_data_int)

    # Step 4: Create a DataFrame and combine all data
    # Target column
    df = pd.DataFrame(target_data, columns=[target_column])

    # Dense features
    dense_column_names = [f'col_{i+1}' for i in range(num_dense)]
    dense_df = pd.DataFrame(dense_data, columns=dense_column_names)

    # Sparse features
    sparse_column_names = [f'col_{i+1+num_dense}' for i in range(num_sparse)]
    sparse_df = pd.DataFrame(sparse_data, columns=sparse_column_names)

    # Concatenate all columns together
    df = pd.concat([df, dense_df, sparse_df], axis=1)

    # Step 5: Write the DataFrame to a Parquet file without compression
    df.to_parquet(output_parquet, engine='pyarrow', compression=None)

    print(f"Pseudo dataset generated with {num_rows} rows, {num_dense} dense features (float32), and {num_sparse} sparse features, saved as {output_parquet}")

# Example usage
output_parquet_path = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"
generate_pseudo_parquet(output_parquet_path)
