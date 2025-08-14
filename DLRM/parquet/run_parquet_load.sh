#!/bin/bash

# Define the num-threads values
NUM_THREADS_VALUES=(1 4 8)
# Define the modulus values
MODULUS_VALUES=(8192 536870912)
# MODULUS_VALUES=(8192)
# Define the number of loops for each function
NUM_LOOPS=3

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="data_utils_${file_name}.py"
    output_file="log_load_${script_name}.log"

    # Clear the contents of the output file
    > "$output_file"

    # Loop through the different combinations of num-threads and modulus values
    for num_threads in "${NUM_THREADS_VALUES[@]}"; do
        for modulus in "${MODULUS_VALUES[@]}"; do

            command="python $script_name --n-jobs $num_threads --modulus $modulus"
            # echo "Run #$i with --n-jobs=$num_threads and --modulus=$modulus" | tee -a "$output_file"
            echo $command | tee -a "$output_file"

            for ((i=1; i<=NUM_LOOPS; i++)); do
                # Run the command with the current combination of num-threads and modulus
                { $command; } 2>&1 | tee -a "$output_file"    
            done

            # Add a newline for clarity between runs
            echo -e "\n" >> "$output_file"
        done
    done
}

# Run the tests with the specified scripts

run_tests "parquet_vocab_small_load"
run_tests "parquet_vocab_large_load"

# run_tests "binary_column_vocab_small_load_chunks"
# run_tests "binary_column_vocab_large_load_chunks"

# run_tests "parquet_load"
# run_tests "parquet_vocab"
# run_tests "parquet_vocab_gen"
# run_tests "parquet_vocab_apply"
