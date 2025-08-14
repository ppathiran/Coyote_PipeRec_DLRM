#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# WARNING: must have compiled PyTorch

# Check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi

cuda_arg="CUDA_VISIBLE_DEVICES=0"

dlrm_pt_bin="python3 ../dlrm_s_pytorch_preproc_fpga.py" 

# Define the mini-batch sizes. Note the mini_batch_size * num_samples * 48 cannot be larger than the max. allocated buffer size on GPU, which is 4 MiB. 
mini_batch_sizes=(4096 8192 16384)

# Define the list of num-workers values to iterate over
num_workers_list=(16)

original_file="/mnt/scratch/ppathiran/data/criteo_10gb.txt"
preprocessed_file="/mnt/scratch/ppathiran/data/kaggleAdDisplayChallenge_processed.npz"

# Loop over each mini-batch size
for batch_size in "${mini_batch_sizes[@]}"; do
    echo "Running with mini-batch size: $batch_size"
    
    # Loop over each num-workers value
    for num_workers in "${num_workers_list[@]}"; do
        echo "Running with num-workers: $num_workers"
        
        # Repeat 4 times for each batch_size and num-workers
        for i in {1..1}; do
            echo "Iteration $i for mini-batch size $batch_size and num-workers $num_workers"
            
            # Command with the current batch size and num-workers
	    #cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=fpga --data-set=kaggle --raw-data-file=$original_file --processed-data-file=$preprocessed_file --loss-function=bce --round-targets=True --learning-rate=0.1 --print-freq=100 --print-time --mini-batch-size=${batch_size} --num-workers=${num_workers} --use-gpu $dlrm_extra_option"

	    cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=fpga --loss-function=bce --round-targets=True --learning-rate=0.1 --print-freq=100 --print-time --mini-batch-size=${batch_size} --num-workers=${num_workers} --use-gpu $dlrm_extra_option"

	    echo $cmd
            eval "$cmd" 2> >(grep -v "ERR:  Failed to send a request" >&2)
            echo "Finished iteration $i for mini-batch size: $batch_size and num-workers: $num_workers"
            echo -e "\n"
        done
    done

    printf '%*s\n' 80 '' | tr ' ' '-'
done

echo "done"
