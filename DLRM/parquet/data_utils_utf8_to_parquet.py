import pandas as pd
import numpy as np

def process_file_to_parquet(datafile, output_parquet):
    # Step 1: Read the file line by line
    X_int = []
    X_cat = []
    y = []

    with open(datafile, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            # Split the line by tabs (assuming tab-delimited data)
            line = line.strip().split("\t")

            # Handle missing values by replacing empty strings or newline with "0"
            line = ['0' if x == "" or x == "\n" else x for x in line]

            # Assuming first column is the label (target)
            y.append(np.int32(line[0]))

            # Continuous features (columns 1 to 13)
            X_int.append(np.array(line[1:14], dtype=np.int32))

            # Categorical features (columns 14 onwards) - left pad with zeros to 8 bytes
            X_cap_padded = [x.zfill(8) for x in line[14:]]
            X_cat.append(X_cap_padded)  # Left pad with zeros
            print("i: ", i, ", ", X_cap_padded)

    # Convert lists to NumPy arrays for continuous and categorical features
    y = np.array(y, dtype=np.int32)
    X_int = np.array(X_int, dtype=np.int32)
    X_cat = np.array(X_cat, dtype='S8')  # Ensure each element is 8 bytes (S8 = 8-byte strings)

    # Step 4: Create a DataFrame from the processed data
    df = pd.DataFrame({
        'y': y,                        # Label
        **{f'X_int_{i}': X_int[:, i] for i in range(13)},   # Continuous features
        **{f'X_cat_{i}': X_cat[:, i].astype(str) for i in range(X_cat.shape[1])}  # Convert byte strings back to str
    })

    # Step 5: Write DataFrame to Parquet file
    df.to_parquet(output_parquet, engine='pyarrow', compression='snappy')

    print(f"File successfully written to {output_parquet}")

# Example usage
datafile = "/local/home/yuzhuyu/train.txt"   # Replace with the actual UTF-8 file path
output_parquet = "/local/home/yuzhuyu/parquet_file.parquet"  # Output Parquet file
process_file_to_parquet(datafile, output_parquet)
