#!/bin/bash

NUM_LOOPS=3

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="operator_test_${file_name}.py"
    output_file="log_${script_name}.log"

    # Clear the contents of the output file
    > "$output_file"

    # for modulus in 8192 536870912; do
    for modulus in 536870912; do
        # Construct the command with the current modulus
        command="python $script_name --n-jobs 1 --modulus $modulus"

        # Log the command
        echo $command | tee -a "$output_file"

        for ((i=1; i<=NUM_LOOPS; i++)); do
            # Run the command with the current modulus value
            { $command; } 2>&1 | tee -a "$output_file"    
        done

        # Add a newline for clarity between modulus values
        echo -e "\n" >> "$output_file"
    done
}

# Run the tests with the specified script
# run_tests "gpu"
# run_tests "dense"
run_tests "sparse"

