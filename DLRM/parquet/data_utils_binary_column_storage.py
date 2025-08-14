import pandas as pd
import numpy as np
import struct

# Path to the Parquet file and output binary file


def convert_parquet_to_binary(parquet_file, output_binary_file, metadata_file):
    # Load the Parquet file
    df = pd.read_parquet(parquet_file)

    # Get the number of rows and columns
    num_rows, num_columns = df.shape

    # Dictionary to store metadata for each column
    metadata = {
        "num_rows": num_rows,
        "num_columns": num_columns,
        "columns": []
    }

    # Write the data to a single binary file
    with open(output_binary_file, 'wb') as binary_file:
        for column in df.columns:
            # Convert the column data to a numpy array
            column_data = df[column].to_numpy()

            # Determine the data type
            dtype = column_data.dtype
            dtype_str = dtype.str  # Get the data type as a string, e.g., '<f8' for float64

            # Save metadata for this column
            metadata["columns"].append({
                "name": column,
                "dtype": dtype_str,
                "offset": binary_file.tell(),  # Current position in the binary file
                "size": column_data.nbytes     # Size in bytes for this column
            })

            # Write the column data as bytes to the binary file
            binary_file.write(column_data.tobytes())

    # Save metadata to a text file for easy access during reads
    with open(metadata_file, 'w') as f:
        f.write(str(metadata))

print("Converting Parquet to Binary for 13 dense and 26 sparse columns")
parquet_file = "/local/home/yuzhuyu/bin2parquet.parquet"
output_binary_file = "/local/home/yuzhuyu/columnar_data_13_dense_26_sparse.bin"
metadata_file = "/local/home/yuzhuyu/columnar_metadata_13_dense_26_sparse.txt"
convert_parquet_to_binary(parquet_file, output_binary_file, metadata_file)

print("Converting Parquet to Binary for 504 dense and 42 sparse columns")
parquet_file = "/local/home/yuzhuyu/pseudo_dataset_504_dense_42_sparse.parquet"
output_binary_file = "/local/home/yuzhuyu/columnar_data_504_dense_42_sparse.bin"
metadata_file = "/local/home/yuzhuyu/columnar_metadata_504_dense_42_sparse.txt"
convert_parquet_to_binary(parquet_file, output_binary_file, metadata_file)
