# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset, data_utils
from copy import deepcopy
from tqdm import tqdm
import pickle


class ConformationSampleDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        seed,
        coordinates,
    ):
        self.dataset = dataset
        self.seed = seed
        self.coordinates = coordinates
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        data = deepcopy(self.dataset[index])
        if "uid" not in data:
            data["uid"] = index
        if not isinstance(data[self.coordinates], list):
            return data
        size = len(data[self.coordinates])
        with data_utils.numpy_seed(self.seed, epoch, index):
            sample_idx = np.random.randint(size)
        coordinates = data[self.coordinates][sample_idx]
        del data[self.coordinates]
        data[self.coordinates] = coordinates
        return data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)


class ConformationExpandDataset(BaseWrapperDataset):
    def __init__(self, dataset, seed, coordinates, repeat_count=1):
        self.dataset = dataset
        self.seed = seed
        self.coordinates = coordinates
        self.repeat_count = repeat_count
        self._init_idx()
        self.set_epoch(None)

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.epoch = epoch

    def _init_idx(self):
        self.idx2key = []
        for i in tqdm(range(len(self.dataset))):
            if isinstance(self.dataset[i][self.coordinates], list):
                size = len(self.dataset[i][self.coordinates])
            else:
                size = 1
            for j in range(size):
                self.idx2key.extend([(i, j) for _ in range(self.repeat_count)])
        self.cnt = len(self.idx2key)

    def __len__(self):
        return self.cnt

    @lru_cache(maxsize=16)
    def __cached_item__(self, index: int, epoch: int):
        key_idx, conf_idx = self.idx2key[index]
        data = self.dataset[key_idx]
        if isinstance(data[self.coordinates], list):
            coordinates = data[self.coordinates][conf_idx]
        else:
            coordinates = data[self.coordinates]
        if "uid" not in data:
            data["uid"] = key_idx

        ret_data = deepcopy(data)
        del ret_data[self.coordinates]
        ret_data[self.coordinates] = coordinates
        return ret_data

    def __getitem__(self, index: int):
        return self.__cached_item__(index, self.epoch)
