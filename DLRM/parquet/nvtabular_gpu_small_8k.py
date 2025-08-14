import time
import cudf
import nvtabular as nvt
from nvtabular.io import Dataset
from nvtabular.ops import Categorify, FillMissing, LogOp, LambdaOp
import pynvml
import threading
import warnings
import argparse  # Import argparse for argument parsing

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="merlin.dtypes.mappings")
warnings.filterwarnings("ignore", category=FutureWarning, message="promote has been superseded by mode='default'")

# Path to your Parquet file
train_file = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"
# train_file = "/home/yuzhuyu/parquet/bin2parquet.parquet"

# Define column groups
CRITEO_CONTINUOUS_COLUMNS = [f'col_{x}' for x in range(1, 14)]
CRITEO_CATEGORICAL_COLUMNS = [f'col_{x}' for x in range(14, 40)]
CRITEO_CLICK_COLUMNS = ['col_0']

MAX_INDEX = 8192

# Define power monitoring function
def monitor_gpu_power(duration, interval):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming single GPU
    for _ in range(duration // interval):
        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
        print(f"GPU Power Draw: {power:.2f} W")
        time.sleep(interval)
    pynvml.nvmlShutdown()

# Function to preprocess the dataset
def preprocess_criteo_parquet(train_file, frequency_threshold, part_mem_fraction):
    # Define transformations
    cat_features = (
        CRITEO_CATEGORICAL_COLUMNS
        >> LambdaOp(lambda col: 
                    # Convert hexadecimal strings to integers if dtype is 'object'
                    col.str.hex_to_int() % MAX_INDEX  # Direct conversion using `.str.hex_to_int()`
                    if col.dtype == 'object' 
                    else col % MAX_INDEX)
        >> Categorify(freq_threshold=frequency_threshold)
    )

    cont_features = (
        CRITEO_CONTINUOUS_COLUMNS 
        >> LambdaOp(lambda col: col.clip(lower=0))
        >> LogOp()
    )

    # Define the workflow
    workflow = nvt.Workflow(cat_features + cont_features + CRITEO_CLICK_COLUMNS)

    # Load dataset with the dynamic part_mem_fraction argument
    train_ds = Dataset(train_file, engine="parquet", part_mem_fraction=part_mem_fraction)

    # Fit and transform the workflow
    print("Fitting the workflow")
    start = time.time()
    workflow.fit(train_ds)
    print(f"Workflow fitting completed in {time.time() - start:.2f} seconds")

    print("Transforming the dataset")
    start = time.time()
    transformed_ds = workflow.transform(train_ds)
    print(f"Dataset transformation completed in {time.time() - start:.2f} seconds")

    return transformed_ds

# Main function
def main():
    start_time = time.time()

    # Parse the argument for part_mem_fraction
    parser = argparse.ArgumentParser(description="NVTabular Preprocessing with Memory Fraction Control")
    parser.add_argument('--part_mem_fraction', type=float, default=0.3, help="Fraction of GPU memory to use for each data partition (0-1)")
    args = parser.parse_args()

    # Preprocess dataset without Dask
    frequency_threshold = 0  # Example frequency threshold
    transformed_data = preprocess_criteo_parquet(train_file, frequency_threshold, args.part_mem_fraction)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

    # Example: Access transformed data in memory
    # for part in transformed_data.to_iter():
    #     print(part.head())  # Example to print the first few rows of each partition

if __name__ == '__main__':
    main()
