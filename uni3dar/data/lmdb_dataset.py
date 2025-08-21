# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import lmdb
import os
import numpy as np
import gzip
import pickle
from functools import lru_cache
import logging


logger = logging.getLogger(__name__)


class LMDBDataset:
    def __init__(
        self,
        db_path,
        key_to_id=False,
        gzip=True,
        sample_cluster=False,
    ):
        self.db_path = db_path
        assert os.path.isfile(self.db_path), "{} not found".format(self.db_path)
        env = self.connect_db(self.db_path)
        with env.begin() as txn:
            self._keys = list(txn.cursor().iternext(values=False))

        self.key_to_id = key_to_id
        self.gzip = gzip
        self.sample_cluster = sample_cluster

    def connect_db(self, lmdb_path, save_to_self=False):
        env = lmdb.open(
            lmdb_path,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
        if not save_to_self:
            return env
        else:
            self.env = env

    def __len__(self):
        return len(self._keys)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if not hasattr(self, "env"):
            self.connect_db(self.db_path, save_to_self=True)
        if not self.sample_cluster:
            key = self._keys[idx]
        else:
            key = idx.to_bytes(4, byteorder="big")
        datapoint_pickled = self.env.begin().get(key)
        if self.gzip:
            datapoint_pickled = gzip.decompress(datapoint_pickled)
        data = pickle.loads(datapoint_pickled)
        if self.key_to_id:
            data["uid"] = int.from_bytes(key, "big")
        return data

    def calc_mean_std(self, key, inner_idx=None):
        sum = 0
        sum_sq = 0
        count = 0
        for i in range(len(self)):
            data = self[i]
            if inner_idx is not None:
                data = data[key][inner_idx]
            else:
                data = data[key]
            data = np.array(data).reshape(-1)
            sum += np.sum(data)
            sum_sq += np.sum(data**2)
            count += len(data)
        mean = sum / count
        std = np.sqrt(sum_sq / count - mean**2)
        return mean, std
