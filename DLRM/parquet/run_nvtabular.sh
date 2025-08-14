#!/bin/bash

NUM_LOOPS=1

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="nvtabular_gpu_${file_name}.py"
    output_file="log_${script_name}.log"

    # Clear the contents of the output file
    > "$output_file"

    command="python $script_name"
    # echo "Run #$i with --n-jobs=$num_threads and --modulus=$modulus" | tee -a "$output_file"
    echo $command | tee -a "$output_file"

    for ((i=1; i<=NUM_LOOPS; i++)); do
        # Run the command with the current combination of num-threads and modulus
        { $command; } 2>&1 | tee -a "$output_file"    
    done

    # Add a newline for clarity between runs
    echo -e "\n" >> "$output_file"

}

# Run the tests with the specified scripts


# run_tests "small_8k_no_vocab"
# run_tests "small_8k"
run_tests "small_512k"
# run_tests "large_8k"
# run_tests "large_512k"
