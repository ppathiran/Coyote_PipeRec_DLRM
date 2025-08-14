# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Description: generate inputs and targets for the DLRM benchmark
#
# Utility function(s) to download and pre-process public data sets
#   - Criteo Kaggle Display Advertising Challenge Dataset
#     https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
#   - Criteo Terabyte Dataset
#     https://labs.criteo.com/2013/12/download-terabyte-click-logs
#
# After downloading dataset, run:
#   getCriteoAdData(
#       datafile="<path-to-train.txt>",
#       o_filename=kaggleAdDisplayChallenge_processed.npz,
#       max_ind_range=-1,
#       sub_sample_rate=0.0,
#       days=7,
#       data_split='train',
#       randomize='total',
#       criteo_kaggle=True,
#       memory_map=False
#   )
#   getCriteoAdData(
#       datafile="<path-to-day_{0,...,23}>",
#       o_filename=terabyte_processed.npz,
#       max_ind_range=-1,
#       sub_sample_rate=0.0,
#       days=24,
#       data_split='train',
#       randomize='total',
#       criteo_kaggle=False,
#       memory_map=False
#   )

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from multiprocessing import Manager, Process

# import os
from os import path

# import io
# from io import StringIO
# import collections as coll

import numpy as np
import time



def loadDataset(
    dataset,
    max_ind_range,
    sub_sample_rate,
    randomize,
    data_split,
    raw_path="",
    pro_data="",
    memory_map=False,
    dataset_multiprocessing=False,
    num_threads=8
):
    # dataset

    days = num_threads
    o_filename = "kaggleAdDisplayChallenge_processed"

    
    print("Num of threads: ", num_threads)
    print("Reading raw data=%s" % (str(raw_path)))
    file = getCriteoAdData(
        raw_path,
        o_filename,
        max_ind_range,
        sub_sample_rate,
        days,
        data_split,
        randomize,
        dataset == "kaggle",
        memory_map,
        dataset_multiprocessing
    )

    return file, days


def getCriteoAdData(
    datafile,
    o_filename,
    max_ind_range=-1,
    sub_sample_rate=0.0,
    days=7,
    data_split="train",
    randomize="total",
    criteo_kaggle=True,
    memory_map=False,
    dataset_multiprocessing=False,
):
    

if __name__ == "__main__":
    ### import packages ###
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(description="Preprocess Criteo dataset")
    # model related parameters
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--dataset-multiprocessing", action="store_true", default=False)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    loadDataset(
        args.data_set,
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.processed_data_file,
        args.memory_map,
        args.dataset_multiprocessing,
        args.num_threads
    )
