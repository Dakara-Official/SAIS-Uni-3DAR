import numpy as np
from functools import lru_cache
from unicore.data import BaseWrapperDataset


class DataTypeDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        type,
    ):
        self.dataset = dataset
        self.type = type

    def __getitem__(self, index: int):
        data = self.dataset[index]
        data["data_type"] = self.type
        return data
