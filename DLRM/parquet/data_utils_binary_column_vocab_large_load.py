from joblib import Parallel, delayed
import numpy as np
import time
import argparse
import struct

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=8, type=int, metavar='N',
                    help='number of total jobs to run')
parser.add_argument('--modulus', default=8192, type=int, metavar='M',
                    help='modulus value for sparse features')

# Path to your binary file
binary_file = "/local/home/yuzhuyu/columnar_data_504_dense_42_sparse.bin"
dense_feature = 504
sparse_feature = 42
num_rows = 4000000  # Replace with the actual number of rows in your binary file

# Define the function to read a specific column from a binary file
def read_column_from_binary(column_index, num_rows, row_size, dtype):
    # Define the struct format for different data types
    dtype_formats = {
        np.int32: 'i',  # 4-byte integer
        np.float32: 'f' # 4-byte float
    }
    
    # Get the correct format string for the specified dtype
    format_string = dtype_formats[dtype]
    data = []
    
    with open(binary_file, "rb") as f:
        for row in range(num_rows):
            # Calculate the byte offset for the specific column in this row
            offset = row * row_size + column_index * dtype().nbytes
            f.seek(offset)
            # Read and unpack the value using the format string
            data.append(struct.unpack(format_string, f.read(dtype().nbytes))[0])
    
    return np.array(data)

# Define your pipeline functions using `read_column_from_binary`
def process_column_in_pipeline_0(column_index, num_rows, row_size):
    # Read the target column from the binary file
    data = read_column_from_binary(column_index, num_rows, row_size, dtype=np.int32)  # Assuming int32 for target
    return data

def process_column_in_pipeline_1(column_index, num_rows, row_size):
    # Read a dense feature column and apply any transformations
    data = read_column_from_binary(column_index, num_rows, row_size, dtype=np.float32)  # Assuming float32 for dense
    return data

def process_column_in_pipeline_2(column_index, num_rows, row_size, modulus):
    # Read a sparse feature column, apply HEX to int conversion and modulus
    data = read_column_from_binary(column_index, num_rows, row_size, dtype=np.int32)  # Assuming int32 for sparse
    # Apply modulus transformation
    data = np.mod(data, modulus)
    return data

# Main function to process in parallel
def main():
    args = parser.parse_args()
    modulus = args.modulus

    # Define parameters for reading binary data
    row_size = (1 * np.int32().nbytes + dense_feature * np.float32().nbytes + sparse_feature * np.int32().nbytes)  # Total row size in bytes

    start_time = time.time()

    # Define the column indices for the pipelines
    columns_pipeline_0 = [0]  # Target column index
    columns_pipeline_1 = list(range(1, (1+dense_feature)))  # Dense feature column indices
    columns_pipeline_2 = list(range((1+dense_feature), (1+dense_feature+sparse_feature)))  # Sparse feature column indices

    result = Parallel(n_jobs=args.n_jobs)(
        [delayed(process_column_in_pipeline_0)(col, num_rows, row_size) for col in columns_pipeline_0] +
        [delayed(process_column_in_pipeline_1)(col, num_rows, row_size) for col in columns_pipeline_1] +  
        [delayed(process_column_in_pipeline_2)(col, num_rows, row_size, modulus) for col in columns_pipeline_2]
    )

    # Record the end time and calculate the total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time:.2f} seconds")

# Entry point for the program
if __name__ == "__main__":
    main()
