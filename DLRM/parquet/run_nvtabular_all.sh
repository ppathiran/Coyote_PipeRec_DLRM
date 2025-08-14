#!/bin/bash

NUM_LOOPS=2

# Predefined values for part_mem_fraction
PART_MEM_FRACTIONS=(0.1 0.2 0.3 0.4 0.5)

# Function to monitor GPU power usage, memory usage, and GPU utilization
monitor_gpu_resources () {
    # Add part_mem_fraction to the log file name
    log_file_utilization="gpu_utilization_${script_name}_part_mem_fraction_${part_mem_fraction}.log"
    
    # Clear the log files
    > "$log_file_utilization"

    # Monitor GPU resources for GPU 0 every second until stopped
    while true; do
        # Record GPU utilization (GPU and memory) for GPU 1
        nvidia-smi -i 1 --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$log_file_utilization"
        
        # Sleep for 0.1 second between updates
        sleep 0.1
    done
}

# Function to run tests
run_tests () {
    # Use the provided script name as the first argument
    file_name="$1"
    script_name="nvtabular_gpu_${file_name}.py"
    
    # Add part_mem_fraction to the log file name
    output_file="log_${script_name}_part_mem_fraction_${part_mem_fraction}.log"

    # Clear the contents of the output file
    > "$output_file"

    # Command to run Python script
    command="python $script_name"

    for part_mem_fraction in "${PART_MEM_FRACTIONS[@]}"; do
        echo "Running with part_mem_fraction=$part_mem_fraction"

        # Add the part_mem_fraction to the command
        command_with_memory="python $script_name --part_mem_fraction $part_mem_fraction"
        echo $command_with_memory | tee -a "$output_file"

        # Start GPU monitoring in the background
        monitor_gpu_resources & 
        monitor_pid=$!

        for ((i=1; i<=NUM_LOOPS; i++)); do
            # Run the command with the current part_mem_fraction value
            { $command_with_memory; } 2>&1 | tee -a "$output_file"    
        done

        # Stop GPU monitoring after the Python script finishes
        kill $monitor_pid
        wait $monitor_pid 2>/dev/null

        # Add a newline for clarity between runs
        echo -e "\n" >> "$output_file"
    done
}

# Run the tests with the specified scripts
run_tests "small_8k_no_vocab"
run_tests "small_8k"
run_tests "small_512k"

run_tests "large_8k_no_vocab"
run_tests "large_8k"
run_tests "large_512k"
