#!/bin/bash

# Define the num-threads values
NUM_THREADS_VALUES=(64)
# Define the modulus values
MODULUS_VALUES=(8192 536870912)
# Define the number of loops for each function
NUM_LOOPS=1

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="data_utils_${file_name}.py"
    output_file="log_${script_name}.log"

    # Clear the contents of the output file
    > "$output_file"

    # Check if the script is one of the ones to run only once
    if [[ "$file_name" == "parquet_vocab_small_no" || "$file_name" == "parquet_vocab_large_no" ]]; then
        # If it's one of the "no" scripts, run only once with all modulus values
        for num_threads in "${NUM_THREADS_VALUES[@]}"; do
            command="python $script_name --n-jobs $num_threads"
            echo $command | tee -a "$output_file"
            for ((i=1; i<=NUM_LOOPS; i++)); do
                { $command; } 2>&1 | tee -a "$output_file"    
            done
            echo -e "\n" >> "$output_file"
        done
    else
        # If it's one of the "gen" scripts, loop through different modulus values
        for num_threads in "${NUM_THREADS_VALUES[@]}"; do
            for modulus in "${MODULUS_VALUES[@]}"; do
                command="python $script_name --n-jobs $num_threads --modulus $modulus"
                echo $command | tee -a "$output_file"
                for ((i=1; i<=NUM_LOOPS; i++)); do
                    { $command; } 2>&1 | tee -a "$output_file"    
                done
                echo -e "\n" >> "$output_file"
            done
        done
    fi
}

# Run tests with the provided script names
run_tests "parquet_vocab_small_no"
run_tests "parquet_vocab_small_gen"

run_tests "parquet_vocab_large_no"
run_tests "parquet_vocab_large_gen"
