#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#check if extra argument is passed to the test
if [[ $# == 1 ]]; then
    dlrm_extra_option=$1
else
    dlrm_extra_option=""
fi
#echo $dlrm_extra_option

cpu=1
gpu=1
pt=1

ncores=6 #28 #12 #6
nsockets="0"

ngpus="1"

# numa_cmd="numactl --physcpubind=0-$((ncores-1)) -m $nsockets" #run on one socket, without HT
dlrm_pt_bin="python dlrm_s_pytorch.py"
# dlrm_c2_bin="python dlrm_s_caffe2.py"

data=random #synthetic
print_freq=100
rand_seed=727

# c2_net="async_scheduling"

#Model param
mb_size=2048 #1024 #512 #256
nbatches=1000 #500 #100
bot_mlp="512-512-64"
top_mlp="1024-1024-1024-1"
emb_size=64
nindices=100
emb="1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
interaction="dot"
tnworkers=0
tmb_size=16384

#_args="--mini-batch-size="${mb_size}\
_args=" --num-batches="${nbatches}\
" --data-generation="${data}\
" --arch-mlp-bot="${bot_mlp}\
" --arch-mlp-top="${top_mlp}\
" --arch-sparse-feature-size="${emb_size}\
" --arch-embedding-size="${emb}\
" --num-indices-per-lookup="${nindices}\
" --arch-interaction-op="${interaction}\
" --numpy-rand-seed="${rand_seed}\
" --print-freq="${print_freq}\
" --print-time"\
" --enable-profiling "

# c2_args=" --caffe2-net-type="${c2_net}


# # CPU Benchmarking
# if [ $cpu = 1 ]; then
#   echo "--------------------------------------------"
#   echo "CPU Benchmarking - running on $ncores cores"
#   echo "--------------------------------------------"
#   if [ $pt = 1 ]; then
#     outf="model1_CPU_PT_$ncores.log"
#     outp="dlrm_s_pytorch.prof"
#     echo "-------------------------------"
#     echo "Running PT (log file: $outf)"
#     echo "-------------------------------"
#     # cmd="$numa_cmd $dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args $dlrm_extra_option > $outf"
#     cmd="$dlrm_pt_bin --mini-batch-size=$mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args $dlrm_extra_option > $outf"
#     echo $cmd
#     eval $cmd
#     min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
#     echo "Min time per iteration = $min"
#     # move profiling file(s)
#     mv $outp ${outf//".log"/".prof"}
#     mv ${outp//".prof"/".json"} ${outf//".log"/".json"}

#   fi
# fi

# GPU Benchmarking
if [ $gpu = 1 ]; then
  echo "--------------------------------------------"
  echo "GPU Benchmarking - running on $ngpus GPUs"
  echo "--------------------------------------------"
  for _ng in $ngpus
  do
    # weak scaling
    # _mb_size=$((mb_size*_ng))
    # strong scaling
    _mb_size=$((mb_size*1))
    _gpus=$(seq -s, 0 $((_ng-1)))
    cuda_arg="CUDA_VISIBLE_DEVICES=$_gpus"
    echo "-------------------"
    echo "Using GPUS: "$_gpus
    echo "-------------------"
    if [ $pt = 1 ]; then
      outf="model1_GPU_PT_$_ng.log"
      outp="dlrm_s_pytorch.prof"
      echo "-------------------------------"
      echo "Running PT (log file: $outf)"
      echo "-------------------------------"
      cmd="$cuda_arg $dlrm_pt_bin --mini-batch-size=$_mb_size --test-mini-batch-size=$tmb_size --test-num-workers=$tnworkers $_args --use-gpu $dlrm_extra_option > $outf"
      # cmd="$cuda_arg $dlrm_pt_bin --arch-sparse-feature-size=16 --arch-mlp-bot="13-512-256-64-16" --arch-mlp-top="512-256-1" --data-generation=dataset --data-set=kaggle --raw-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/train.txt --processed-data-file=/home/yuzhuyu/u55c/criteo_sharded/train_1/kaggleAdDisplayChallenge_processed.npz --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1024 --print-time --test-mini-batch-size=16384 --test-num-workers=16 --use-gpu $dlrm_extra_option 2>&1 | tee $outf"
      echo $cmd
      eval $cmd
      min=$(grep "iteration" $outf | awk 'BEGIN{best=999999} {if (best > $7) best=$7} END{print best}')
      echo "Min time per iteration = $min"
      # move profiling file(s)
      mv $outp ${outf//".log"/".prof"}
      mv ${outp//".prof"/".json"} ${outf//".log"/".json"}
    fi
  done
fi
