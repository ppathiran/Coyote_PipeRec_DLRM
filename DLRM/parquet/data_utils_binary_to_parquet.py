import numpy as np
import pandas as pd

def binary_to_parquet_partial(binary_file, output_parquet, total_columns=48, selected_columns=40, dtype=np.int32):
    # Step 1: Read the binary file into a NumPy array
    # Each column is 32 bits (4 bytes), so dtype=np.int32
    with open(binary_file, 'rb') as f:
        # Read all the data from the binary file
        data = np.fromfile(f, dtype=dtype)

    # Step 2: Reshape the data to match the expected number of total columns
    if data.size % total_columns != 0:
        raise ValueError("The data size is not divisible by the total number of columns.")
    data = data.reshape(-1, total_columns)

    # Step 3: Select only the first `selected_columns` (e.g., 40 out of 48)
    selected_data = data[:, :selected_columns]

    # Step 4: Create a Pandas DataFrame from the selected columns
    # Create column names like 'col_0', 'col_1', ..., 'col_39' for 40 columns
    column_names = [f'col_{i}' for i in range(selected_columns)]
    df = pd.DataFrame(selected_data, columns=column_names)

    # Step 5: Write the DataFrame to a Parquet file
    df.to_parquet(output_parquet, engine='pyarrow', compression='snappy')

    print(f"First {selected_columns} columns of binary file converted to Parquet and saved as {output_parquet}")

# Example usage
binary_file_path = "/local/home/yuzhuyu/processed_data.bin"
output_parquet_path = "/local/home/yuzhuyu/bin2parquet.parquet"

# Call the function to convert binary to Parquet
binary_to_parquet_partial(binary_file_path, output_parquet_path)
