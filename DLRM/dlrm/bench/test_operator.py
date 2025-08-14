from os import path
import numpy as np

import time

datafile = '/local/home/yuzhuyu/train.txt'

#! Read data into array
days = 1
total_count = 0
total_per_file = []

t0 = time.perf_counter()
with open(str(datafile)) as f:
    for _ in f:
        total_count += 1
total_per_file.append(total_count)
# reset total per file due to split
num_data_per_split, extras = divmod(total_count, days)
total_per_file = [num_data_per_split] * days
for j in range(extras):
    total_per_file[j] += 1
# split into days (simplifies code later on)

file_id = 0
boundary = total_per_file[file_id]

sif_output = [[] for _ in range(days)]
current_file_data = sif_output[file_id]
with open(str(datafile)) as f:
    for j, line in enumerate(f):
        if j == boundary:
            file_id += 1
            # MODIFIED: Switch to the next inner list for the new "file" or split of data.
            current_file_data = sif_output[file_id]
            boundary += total_per_file[file_id]
        # MODIFIED: Append line to the in-memory data structure instead of writing to a file.
        current_file_data.append(line)

t1 = time.perf_counter()
print(f"Number of rows: {total_count}")
print(f"Count the number of rows: {t1-t0} s")