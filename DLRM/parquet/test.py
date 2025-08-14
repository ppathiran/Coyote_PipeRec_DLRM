# import pyarrow.parquet as pq

# # Path to your Parquet file
# # parquet_file = "/mnt/scratch/yuzhuyu/parquet/bin2parquet.parquet"
# parquet_file = "/mnt/scratch/yuzhuyu/parquet/pseudo_dataset_504_dense_42_sparse.parquet"

# # Read the Parquet file metadata
# parquet_file_metadata = pq.read_metadata(parquet_file)

# # Print the schema of the Parquet file
# schema = parquet_file_metadata.schema
# print("Schema of the Parquet file:")
# print(schema)

import pyarrow.parquet as pq
import pyarrow as pa

# Path to your Parquet file
parquet_file = "/mnt/scratch/yuzhuyu/parquet/pseudo_dataset_504_dense_42_sparse.parquet_no_compress"

# Function to determine bit width for each data type
def get_bit_width(value):
    if isinstance(value, int):
        # Assume 32-bit for int (int32)
        return 32
    elif isinstance(value, float):
        # Assume 32-bit for float (float32)
        return 32
    elif isinstance(value, bool):
        # Booleans are stored as 1 byte (8 bits)
        return 8
    elif isinstance(value, str):
        # For string, we return the length of the string in bytes (8 bits per character)
        return len(value) * 8
    elif isinstance(value, bytes):
        # For binary data, same as string
        return len(value) * 8
    else:
        return "Unknown Type"  # For other types like list, map, etc.

# Read the Parquet file
table = pq.read_table(parquet_file)

# Get the first row of the table
first_row = table.slice(0, 1).to_pandas()

# Iterate over each column and check the bit width of the first row's element
print("Width of each element in the first row:")
for column in first_row.columns:
    value = first_row[column].iloc[0]  # Get the first element of the column
    bit_width = get_bit_width(value)
    print(f"Column: {column}, First Element: {value}, Bit Width: {bit_width} bits")
