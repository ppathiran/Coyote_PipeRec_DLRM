from multiprocessing import Manager, Process
from os import path

import time
import numpy as np


datafile = '/home/yuzhuyu/u55c/criteo_sharded/train_1/train.txt'

## Test pure I/O speed
total_count = 0
total_per_file = []

if path.exists(datafile):

    t1_0 = time.perf_counter()
    print("Reading data from path=%s" % (datafile))
    
    with open(str(datafile)) as f:
        for _ in f:
            total_count += 1
    total_per_file.append(total_count)
    t1_1 = time.perf_counter()

    file_size = path.getsize(datafile)
    file_size_GB = file_size / 1024 / 1024 / 1024
    execution_time = t1_1 - t1_0
    print(f"The size of '{datafile}' is {file_size} B.")
    print(f"The size of '{datafile}' is {file_size_GB} GB.")
    print("Task 1 Execution time: %s s", (execution_time))
    print("throughput: ", (file_size_GB / execution_time), "GB/s")
    print("")


## Test the speed of spliting files
days = 7
criteo_kaggle = True

lstr = datafile.split("/")
d_path = "/".join(lstr[0:-1]) + "/"
d_file = lstr[-1].split(".")[0] if criteo_kaggle else lstr[-1]
npzfile = d_path + ((d_file + "_day") if criteo_kaggle else d_file)
trafile = d_path + ((d_file + "_fea") if criteo_kaggle else "fea")

if path.exists(datafile):

    t2_0 = time.perf_counter()
    print("Reading data from path=%s" % (datafile))
    # with open(str(datafile)) as f:
    #     for _ in f:
    #         total_count += 1
    # total_per_file.append(total_count)
    # reset total per file due to split
    num_data_per_split, extras = divmod(total_count, days)
    total_per_file = [num_data_per_split] * days
    for j in range(extras):
        total_per_file[j] += 1
    # split into days (simplifies code later on)
    file_id = 0
    boundary = total_per_file[file_id]
    nf = open(npzfile + "_" + str(file_id), "w")
    with open(str(datafile)) as f:
        for j, line in enumerate(f):
            if j == boundary:
                nf.close()
                file_id += 1
                nf = open(npzfile + "_" + str(file_id), "w")
                boundary += total_per_file[file_id]
            nf.write(line)
    nf.close()
    t2_1 = time.perf_counter()
    print("Task 2 Execution time: %s s", (t2_1-t2_0))
    print("")


total_count = 0
if path.exists(datafile):

    t3_0 = time.perf_counter()
    print("Reading data from path=%s" % (datafile))
    
    with open(str(datafile)) as f:
        for k, line in enumerate(f):
            line = line.split("\t")
            for j in range(len(line)):
                if (line[j] == "") or (line[j] == "\n"):
                    line[j] = "0"
    t3_1 = time.perf_counter()

    file_size = path.getsize(datafile)
    file_size_GB = file_size / 1024 / 1024 / 1024
    print(f"The size of '{datafile}' is {file_size} B.")
    print(f"The size of '{datafile}' is {file_size_GB} GB.")
    print("Task 3 Execution time: %s s", (t3_1-t3_0))
    print("throughput: ", (file_size_GB / execution_time), "GB/s")
    print("")




## Test the speed of process_one_file()
sub_sample_rate=0.0
max_ind_range=5000



def process_one_file(
    datfile,
    npzfile,
    split,
    num_data_in_split,
    dataset_multiprocessing,
    convertDictsDay=None,
    resultDay=None,
):
    if dataset_multiprocessing:
        convertDicts_day = [{} for _ in range(26)]
    with open(str(datfile)) as f:
        y = np.zeros(num_data_in_split, dtype="i4")  # 4 byte int
        X_int = np.zeros((num_data_in_split, 13), dtype="i4")  # 4 byte int
        X_cat = np.zeros((num_data_in_split, 26), dtype="i4")  # 4 byte int
        if sub_sample_rate == 0.0:
            rand_u = 1.0
        else:
            rand_u = np.random.uniform(low=0.0, high=1.0, size=num_data_in_split)
        i = 0
        percent = 0
        for k, line in enumerate(f):
            # process a line (data point)
            line = line.split("\t")
            # set missing values to zero
            for j in range(len(line)):
                if (line[j] == "") or (line[j] == "\n"):
                    line[j] = "0"
            # sub-sample data by dropping zero targets, if needed
            target = np.int32(line[0])
            if (
                target == 0
                and (rand_u if sub_sample_rate == 0.0 else rand_u[k])
                < sub_sample_rate
            ):
                continue
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
            # count uniques
            if dataset_multiprocessing:
                for j in range(26):
                    convertDicts_day[j][X_cat[i][j]] = 1
                # debug prints
                if float(i) / num_data_in_split * 100 > percent + 1:
                    percent = int(float(i) / num_data_in_split * 100)
                    print(
                        "Load %d/%d (%d%%) Split: %d  Label True: %d  Stored: %d"
                        % (
                            i,
                            num_data_in_split,
                            percent,
                            split,
                            target,
                            y[i],
                        ),
                        end="\n",
                    )
            else:
                for j in range(26):
                    convertDicts[j][X_cat[i][j]] = 1
                # debug prints
                print(
                    "Load %d/%d  Split: %d  Label True: %d  Stored: %d"
                    % (
                        i,
                        num_data_in_split,
                        split,
                        target,
                        y[i],
                    ),
                    end="\r",
                )
            i += 1
        # store num_data_in_split samples or extras at the end of file
        # count uniques
        # X_cat_t  = np.transpose(X_cat)
        # for j in range(26):
        #     for x in X_cat_t[j,:]:
        #         convertDicts[j][x] = 1
        # store parsed
        filename_s = npzfile + "_{0}.npz".format(split)
        if path.exists(filename_s):
            print("\nSkip existing " + filename_s)
        else:
            np.savez_compressed(
                filename_s,
                X_int=X_int[0:i, :],
                # X_cat=X_cat[0:i, :],
                X_cat_t=np.transpose(X_cat[0:i, :]),  # transpose of the data
                y=y[0:i],
            )
            print("\nSaved " + npzfile + "_{0}.npz!".format(split))
    if dataset_multiprocessing:
        resultDay[split] = i
        convertDictsDay[split] = convertDicts_day
        return
    else:
        return i

dataset_multiprocessing=False
t4_0 = time.perf_counter()
# create all splits (reuse existing files if possible)
recreate_flag = False
convertDicts = [{} for _ in range(26)]
# WARNING: to get reproducable sub-sampling results you must reset the seed below
# np.random.seed(123)
# in this case there is a single split in each day

# for i in range(days):
#     npzfile_i = npzfile + "_{0}.npz".format(i)
#     npzfile_p = npzfile + "_{0}_processed.npz".format(i)
#     if path.exists(npzfile_i):
#         print("Skip existing " + npzfile_i)
#     elif path.exists(npzfile_p):
#         print("Skip existing " + npzfile_p)
#     else:
#         recreate_flag = True
recreate_flag = True
if recreate_flag:
    if dataset_multiprocessing:
        resultDay = Manager().dict()
        convertDictsDay = Manager().dict()
        processes = [
            Process(
                target=process_one_file,
                name="process_one_file:%i" % i,
                args=(
                    npzfile + "_{0}".format(i),
                    npzfile,
                    i,
                    total_per_file[i],
                    dataset_multiprocessing,
                    convertDictsDay,
                    resultDay,
                ),
            )
            for i in range(0, days)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        for day in range(days):
            total_per_file[day] = resultDay[day]
            print("Constructing convertDicts Split: {}".format(day))
            convertDicts_tmp = convertDictsDay[day]
            for i in range(26):
                for j in convertDicts_tmp[i]:
                    convertDicts[i][j] = 1
    else:
        for i in range(days):
            total_per_file[i] = process_one_file(
                npzfile + "_{0}".format(i),
                npzfile,
                i,
                total_per_file[i],
                dataset_multiprocessing,
            )

t4_1 = time.perf_counter()
print("Task 4 Execution time: %s s", (t4_1-t4_0))
print("")

dataset_multiprocessing=False
t4_0 = time.perf_counter()
# create all splits (reuse existing files if possible)
recreate_flag = False
convertDicts = [{} for _ in range(26)]
# WARNING: to get reproducable sub-sampling results you must reset the seed below
# np.random.seed(123)
# in this case there is a single split in each day

# for i in range(days):
#     npzfile_i = npzfile + "_{0}.npz".format(i)
#     npzfile_p = npzfile + "_{0}_processed.npz".format(i)
#     if path.exists(npzfile_i):
#         print("Skip existing " + npzfile_i)
#     elif path.exists(npzfile_p):
#         print("Skip existing " + npzfile_p)
#     else:
#         recreate_flag = True
recreate_flag = True
if recreate_flag:
    if dataset_multiprocessing:
        resultDay = Manager().dict()
        convertDictsDay = Manager().dict()
        processes = [
            Process(
                target=process_one_file,
                name="process_one_file:%i" % i,
                args=(
                    npzfile + "_{0}".format(i),
                    npzfile,
                    i,
                    total_per_file[i],
                    dataset_multiprocessing,
                    convertDictsDay,
                    resultDay,
                ),
            )
            for i in range(0, days)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join()
        for day in range(days):
            total_per_file[day] = resultDay[day]
            print("Constructing convertDicts Split: {}".format(day))
            convertDicts_tmp = convertDictsDay[day]
            for i in range(26):
                for j in convertDicts_tmp[i]:
                    convertDicts[i][j] = 1
    else:
        for i in range(days):
            total_per_file[i] = process_one_file(
                npzfile + "_{0}".format(i),
                npzfile,
                i,
                total_per_file[i],
                dataset_multiprocessing,
            )

t4_1 = time.perf_counter()
print("Task 4 Execution time: %s s", (t4_1-t4_0))
print("")