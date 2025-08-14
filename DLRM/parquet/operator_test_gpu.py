import time
import cudf
import nvtabular as nvt
from nvtabular.io import Dataset
from nvtabular.ops import Categorify, LambdaOp, LogOp
import warnings
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Benchmark for rec_preprocessing')
parser.add_argument('--n-jobs', default=8, type=int, metavar='N',
                    help='number of total jobs to run')
parser.add_argument('--modulus', default=8192, type=int, metavar='M',
                    help='modulus value for sparse features')

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="merlin.dtypes.mappings")
warnings.filterwarnings("ignore", category=FutureWarning, message="promote has been superseded by mode='default'")


# Function to preprocess the dataset (with fit() before transform())
def preprocess_criteo_parquet(train_ds, operator, frequency_threshold, modulus, columns, operator_name):
    
    # Create ColumnSelector for the specified columns (for continuous or categorical)
    column_selector = nvt.ops.ColumnSelector(columns)

    # Create the operator list: column selection and then the operator to apply
    operator_list = column_selector >> operator

    
    # Create the workflow with the selected operator
    workflow = nvt.Workflow(operator_list)

    # If the operator requires fit(), apply it before transform()
    if hasattr(operator, 'fit'):
        start_time = time.time() 
        workflow.fit(train_ds)  # Fit the operator before transform
        end_time = time.time()
        print(f"Operator {operator_name} fitted in {end_time - start_time:.5f} seconds")
    
    start_time = time.time()
    # Apply the operator to the dataset using the workflow
    transformed_data = workflow.transform(train_ds)
    end_time = time.time()
    print(f"Operator {operator_name} transformed in {end_time - start_time:.5f} seconds")

    return transformed_data

# Main function
def main():
    args = parser.parse_args()
    modulus = args.modulus
    frequency_threshold = 0  # Example frequency threshold

    # Define column groups
    CRITEO_CONTINUOUS_COLUMNS = [f'col_{x}' for x in range(1, 14)]
    CRITEO_CATEGORICAL_COLUMNS = [f'col_{x}' for x in range(14, 40)]

    # Define operators for sparse and dense columns
    operators_sparse = [
        LambdaOp(lambda col: col.str.hex_to_int() if col.dtype == 'object' else col),  # Hex to int and modulus
        LambdaOp(lambda col: col % modulus),
        Categorify(freq_threshold=frequency_threshold),  # Categorify
    ]
    
    operators_dense = [
        LambdaOp(lambda col: col.clip(lower=0)),  # Clip negative values for continuous columns
        LogOp()  # Log transform
    ]

    input_file = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"
    # input_file = "/home/yuzhuyu/parquet/bin2parquet.parquet"

    train_ds_sparse = Dataset(input_file, engine="parquet") 
    # Process sparse files sequentially, applying one operator at a time and ensuring only categorical columns are used
    train_ds_sparse = preprocess_criteo_parquet(train_ds_sparse, operators_sparse[0], frequency_threshold, modulus, CRITEO_CATEGORICAL_COLUMNS, "HEX to INT")
    train_ds_sparse = preprocess_criteo_parquet(train_ds_sparse, operators_sparse[1], frequency_threshold, modulus, CRITEO_CATEGORICAL_COLUMNS, "Modulus")
    train_ds_sparse = preprocess_criteo_parquet(train_ds_sparse, operators_sparse[2], frequency_threshold, modulus, CRITEO_CATEGORICAL_COLUMNS, "Categorify")

    train_ds_dense = Dataset(input_file, engine="parquet")
    # Process dense files sequentially, applying one operator at a time and ensuring only continuous columns are used
    train_ds_dense = preprocess_criteo_parquet(train_ds_dense, operators_dense[0], frequency_threshold, modulus, CRITEO_CONTINUOUS_COLUMNS, "Negative to Zero")
    train_ds_dense = preprocess_criteo_parquet(train_ds_dense, operators_dense[1], frequency_threshold, modulus, CRITEO_CONTINUOUS_COLUMNS, "Log Transform")
   

if __name__ == '__main__':
    main()
