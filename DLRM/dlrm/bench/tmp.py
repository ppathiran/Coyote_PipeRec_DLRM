import numpy as np

path = "/mnt/scratch/yuzhuyu/criteo_sharded/train_1/kaggleAdDisplayChallenge_processed.npz"

with np.load(path) as data:
    print("Available keys:", list(data.keys()))

