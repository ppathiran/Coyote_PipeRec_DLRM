# PipeRec
Preprocessing Pipelines for Recommender Models
1. This project is based on Meta's DLRM model. Please refer to this git repo for the original code: https://github.com/facebookresearch/dlrm. The `dlrm` folder in the current repo is modified.
2. The `parquet` folder contains the preprocessing pipelines to handle pseudo Parquet files in the CPU side.

Below are explanations of how to run different files.
1. In dlrm folder, `data_utils.py` is used to generate the preprocessed file in CPU and `data_s_pytorch.py` is used to train DLRM model in GPU.
2. You can download the original `train.txt` here: https://www.kaggle.com/datasets/mrkmakr/criteo-dataset
3. Example code to generate the prerpocessed file from the original text file: `python data_utils.py --max-ind-range 1000000 --raw-data-file ./train.txt --dataset-multiprocessing`, the generated preprocessed file is named as `kaggleAdDisplayChallenge_processed.npz`
4. Example code to run preprocessing pipelines for Parquet (column processing): refer to `parquet/run_parquet.sh`. The parquet folder also contains files to generate the pseudo dataset, to measure the speed of data loading, the execution of single pipeline, the execution of the whole pipeline w/wo vocabularies. Please compare these files to understand its meaning.
5. Example code to train in Nvidia 3090 (with the attached conda environment file): refer to `dlrm/bench/dlrm_s_criteo_kaggle_3090.sh`
