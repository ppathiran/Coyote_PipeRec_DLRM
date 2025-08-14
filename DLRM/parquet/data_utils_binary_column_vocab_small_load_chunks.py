from joblib import Parallel, delayed
import time
import argparse

# Argument parser
parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=8, type=int, metavar='N',
                    help='number of total jobs to run')
parser.add_argument('--modulus', default=8192, type=int, metavar='M',
                    help='modulus value for sparse features')

# Path to your binary file
binary_file = "/local/home/yuzhuyu/columnar_data_13_dense_26_sparse.bin"
num_rows = 45840617  # Replace with the actual number of rows in your binary file
dense_feature = 13
sparse_feature = 26

# Define row size for the binary data
row_size = (1 * 4 + dense_feature * 4 + sparse_feature * 4)  # 4 bytes for each int32 or float32 entry

# Define the function to read a chunk of data from the binary file
def read_binary_chunk(start_row, end_row, row_size):
    with open(binary_file, "rb") as f:
        # Seek to the start of the chunk
        f.seek(start_row * row_size)
        # Read the specified number of rows as one large chunk
        f.read((end_row - start_row) * row_size)

# Main function to process in parallel
def main():
    args = parser.parse_args()
    n_jobs = args.n_jobs

    # Split the work into chunks based on the number of jobs
    rows_per_job = num_rows // n_jobs
    start_time = time.time()

    # Run parallel jobs to read the file in chunks
    Parallel(n_jobs=n_jobs)(
        delayed(read_binary_chunk)(i * rows_per_job, (i + 1) * rows_per_job, row_size)
        for i in range(n_jobs)
    )

    # Record the end time and calculate the total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time for loading: {total_time:.2f} seconds")

# Entry point for the program
if __name__ == "__main__":
    main()
