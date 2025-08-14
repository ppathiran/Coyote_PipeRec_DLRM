import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def generate_pseudo_parquet_with_fixed_length(output_parquet, num_rows=4000000, target_column='col_0', num_dense=504, num_sparse=42):
    # Generate target column
    target_data = np.random.randint(0, 2, size=num_rows)

    # Generate dense features
    dense_data = np.random.uniform(-1000, 1000, size=(num_rows, num_dense)).astype(np.float32)

    # Generate sparse features as fixed-length 8-byte strings
    sparse_data = np.random.randint(0, 0xFFFFFFFF, size=(num_rows, num_sparse), dtype=np.uint32)
    sparse_data_hex = np.vectorize(lambda x: f"{x:08X}")(sparse_data)  # Convert to fixed-length 8-char strings

    # Create Arrow Table with fixed-length types
    fields = [pa.field(target_column, pa.int8())]
    fields += [pa.field(f'col_{i+1}', pa.float32()) for i in range(num_dense)]
    fields += [pa.field(f'col_{i+1+num_dense}', pa.string()) for i in range(num_sparse)]  # Use string type for fixed-length strings

    schema = pa.schema(fields)

    # Build Arrow Table
    target_array = pa.array(target_data, type=pa.int8())
    dense_arrays = [pa.array(dense_data[:, i], type=pa.float32()) for i in range(num_dense)]
    # sparse_arrays = [pa.array(sparse_data_hex[:, i], type=pa.string()) for i in range(num_sparse)]  # Use string arrays
    sparse_arrays = [
        pa.array(sparse_data_hex[:, i].astype("S8"), type=pa.fixed_size_binary(8))
        for i in range(num_sparse)
    ]

    table = pa.Table.from_arrays([target_array] + dense_arrays + sparse_arrays, schema=schema)

    # Write to Parquet
    pq.write_table(table, output_parquet, compression=None)
    print(f"Pseudo dataset with fixed-length sparse features saved to {output_parquet}")

# Example usage
output_parquet_path = "/mnt/scratch/yuzhuyu/parquet/pseudo_dataset_504_dense_42_sparse_fixed_length.parquet"
generate_pseudo_parquet_with_fixed_length(output_parquet_path)
