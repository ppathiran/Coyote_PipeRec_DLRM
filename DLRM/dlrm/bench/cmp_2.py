#! Full decoding
from os import path
import numpy as np

import time

datafile = '/local/home/yuzhuyu/train.txt'
#! Read data into array
days = 1
max_ind_range = 5000
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

data_input = sif_output[0]
num_data_in_split = total_per_file[0]
data_output = []

y = np.zeros(num_data_in_split, dtype="i4") 
X_int = np.zeros((num_data_in_split, 13), dtype="i4")
X_cat = np.zeros((num_data_in_split, 26), dtype="i4") 

i = 0

for k, line in enumerate(data_input):
    # split 
    line = line.split("\t")
    # fill missing vlaue
    for j in range(len(line)):
        if (line[j] == "") or (line[j] == "\n"):
            line[j] = "0"

    # data_output.append(split_line)
    target = np.int32(line[0])

    y[i] = target
    X_int[i] = np.array(line[1:14], dtype=np.int32)
    if max_ind_range > 0:
        X_cat[i] = np.array(
            list(map(lambda x: int(x, 16) % max_ind_range, line[14:])),
            dtype=np.int32,
        )
    else:
        X_cat[i] = np.array(
            list(map(lambda x: int(x, 16), line[14:])), dtype=np.int32
        )
    i += 1

t2 = time.perf_counter()

print(f"Count the number of rows: {t1-t0} s")
print(f"Fill missing values: {t2-t1} s")
print(f"Total: {t2-t0} s")