# -*- coding:utf-8 -*-
import numpy as np


def get_kfold_data(datasets, k=5):
    fold_size = len(datasets) // k
    k_fold_data = []
    for i in range(k):
        val_start = i * fold_size
        if i != k - 1:
            val_end = (i + 1) * fold_size
            validset = datasets[val_start:val_end]
            trainset = datasets[0:val_start] + datasets[val_end:]
        else:
            validset = datasets[val_start:]
            trainset = datasets[0:val_start]
        k_fold_data.append((trainset, validset))
    return k_fold_data


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

