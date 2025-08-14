#!/bin/bash

NUM_LOOPS=1

# Function to monitor GPU power usage
monitor_gpu_power () {
    log_file="gpu_power_${script_name}.log"
    > "$log_file"  # Clear the log file

    # Monitor GPU power for GPU 0 every second until stopped
    while true; do
        nvidia-smi -i 0 --query-gpu=power.draw --format=csv,noheader,nounits >> "$log_file"
        sleep 0.1
    done
}

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="nvtabular_gpu_${file_name}.py"
    output_file="log_${script_name}.log"

    # Clear the contents of the output file
    > "$output_file"

    command="python $script_name"
    echo $command | tee -a "$output_file"

    # Start GPU monitoring in the background
    monitor_gpu_power &
    monitor_pid=$!

    for ((i=1; i<=NUM_LOOPS; i++)); do
        # Run the command with the current combination of num-threads and modulus
        { $command; } 2>&1 | tee -a "$output_file"    
    done

    # Stop GPU monitoring after the Python script finishes
    kill $monitor_pid
    wait $monitor_pid 2>/dev/null

    # Add a newline for clarity between runs
    echo -e "\n" >> "$output_file"
}

# Run the tests with the specified scripts
run_tests "small_8k_no_vocab"
# run_tests "small_8k"
# run_tests "small_512k"
# run_tests "large_8k"
# run_tests "large_512k"
