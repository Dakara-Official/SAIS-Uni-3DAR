# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import torch


import numpy as np
from unicore.data import NestedDictionaryDataset, EpochShuffleDataset
from uni3dar.data import (
    LMDBDataset,
    GridDataset,
    ConformationSampleDataset,
    ConformationExpandDataset,
    DataTypeDataset,
    AtomFeatDict,
    SampleClusterDataset,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


@register_task("uni3dar")
class Uni3DAR(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument(
            "--gzip",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--grid-len",
            default=0.24,
            type=float,
        )
        parser.add_argument(
            "--xyz-resolution",
            default=0.01,
            type=float,
        )
        parser.add_argument(
            "--random-rotation-prob",
            default=1.0,
            type=float,
        )
        parser.add_argument(
            "--merge-level",
            default=7,
            type=int,
        )
        parser.add_argument(
            "--max-num-atom",
            default=-1,
            type=int,
            help="set -1 to disable, otherwise will have additional embedding for atom count prediction",
        )
        parser.add_argument(
            "--tree-delete-ratio",
            default=0.0,
            type=float,
        )
        parser.add_argument(
            "--tree-delete-start-layer",
            default=1,
            type=int,
        )
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--enable-cutoff",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--H-prob",
            default=0.5,
            type=float,
        )
        parser.add_argument(
            "--expand-valid-dataset",
            default=False,
            action="store_true",
        )
        # 新增参数，控制训练模型时，条件特征的随机丢弃阈值
        parser.add_argument(
            "--molecule-energy-cond-drop",
            default=0.1,
            type=float,
        )
        # 新增参数，数据集中最小的properties，用于归一化
        parser.add_argument(
            "--energy-min-stat",
            default=0.1,
            type=float,
        )
        # 新增参数，数据集中最大的properties，用于归一化
        parser.add_argument(
            "--energy-max-stat",
            default=0.1,
            type=float,
        )
        # 新增参数，规定归一化边界的扩展比例
        parser.add_argument(
            "--energy-padding",
            default=0.05,
            type=float,
        )
        # 新增参数，为归一化后条件引入的噪声的最大绝对值
        parser.add_argument(
            "--energy-condition-noise",
            default=0.1,
            type=float,
        )
        # 新增参数，用于指定条件集路径
        parser.add_argument(
            "--condition-file",
            default=None,
            metavar='FILE',
            help='path(s) to condition file(s)'
        )
        parser.add_argument(
            "--data-type",
            default="molecule_energy",
            type=str,
            choices=[
                "molecule_energy",
            ],
        ) # 只保留了我们需要的选项
        parser.add_argument(
            "--atom-type-key",
            default="atom_type",
            type=str,
        )
        parser.add_argument(
            "--atom-pos-key",
            default="atom_pos",
            type=str,
        )
        parser.add_argument(
            "--lattice-matrix-key",
            default="lattice_matrix",
            type=str,
        )
        # 以下参数不需要
        # parser.add_argument(
        #     "--crystal-cond-drop",
        #     default=0.1,
        #     type=float,
        # )
        # parser.add_argument(
        #     "--crystal-pxrd",
        #     default=0,
        #     type=int,
        # )
        # parser.add_argument(
        #     "--crystal-pxrd-step",
        #     default=0.1,
        #     type=float,
        # )
        # parser.add_argument(
        #     "--crystal-pxrd-num-fill",
        #     default=64,
        #     type=int,
        # )
        # parser.add_argument(
        #     "--crystal-pxrd-sqrt",
        #     default=False,
        #     action="store_true",
        # )
        # parser.add_argument(
        #     "--crystal-pxrd-threshold",
        #     default=5,
        #     type=float,
        # )
        # parser.add_argument(
        #     "--crystal-pxrd-noise",
        #     default=0.1,
        #     type=float,
        # )
        # parser.add_argument(
        #     "--crystal-component",
        #     default=0,
        #     type=int,
        # )
        # parser.add_argument(
        #     "--crystal-component-sqrt",
        #     default=False,
        #     action="store_true",
        # )
        # parser.add_argument(
        #     "--crystal-component-noise",
        #     default=0.1,
        #     type=float,
        # )
        # parser.add_argument(
        #     "--crystal-random-shift-prob",
        #     default=1.0,
        #     type=float,
        # )
        parser.add_argument(
            "--repeat-count",
            default=5,
            type=int,
        )
        parser.add_argument(
            "--sample-cluster",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--train-cluster-path",
            type=str,
        )
        parser.add_argument(
            "--valid-cluster-path",
            type=str,
        )
        # parameters for inference
        parser.add_argument(
            "--tree-temperature",
            default="1.0",
            type=str,
        )
        parser.add_argument(
            "--count-temperature",
            default="1.0",
            type=str,
        )
        parser.add_argument(
            "--atom-temperature",
            default="1.0",
            type=str,
        )
        parser.add_argument(
            "--xyz-temperature",
            default="1.0",
            type=str,
        )
        parser.add_argument(
            "--allow-atoms",
            default="H,C,N,O,F,P,S,Cl,Br",
            type=str,
        ) # 保险起见，将默认元素改成比赛涉及的9种元素
        parser.add_argument(
            "--save-path",
            default=None,
            type=str,
        )
        # 以下参数不需要
        # parser.add_argument(
        #     "--num-samples",
        #     default=10000,
        #     type=int,
        # )
        # parser.add_argument(
        #     "--rank-ratio",
        #     default=1.0,
        #     type=float,
        # )
        parser.add_argument(
            "--target-tree-index",
            default=2,
            type=int,
        )
        parser.add_argument(
            "--target-space-index",
            default=0,
            type=int,
        )
        parser.add_argument(
            "--rank-by",
            default="atom",
            type=str,
            choices=["atom", "atom+xyz"],
        )
        parser.add_argument(
            "--res-type-key",
            default="res_type",
            type=str,
        )
        parser.add_argument(
            "--mol-target-key",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--atom-target-key",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--mol-target-idx",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--atom-target-idx",
            type=int,
            default=None,
        )
        parser.add_argument(
            "--mol-num-classes",
            type=int,
            default=1,
        )
        parser.add_argument(
            "--atom-num-classes",
            type=int,
            default=1,
        )
        parser.add_argument("--target-atom-pos-key", type=str, default=None)
        parser.add_argument(
            "--mol-target-normalize",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--atom-target-normalize",
            default=False,
            action="store_true",
        )
        parser.add_argument(
            "--atom-target-rotate",
            default=False,
            action="store_true",
        )

    def __init__(self, args):
        super().__init__(args)
        self.dictionary = AtomFeatDict(args)
        self.seed = args.seed
        self.stats = {}

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args)

    def load_dataset(self, split, **kwargs):
        split_path = os.path.join(self.args.data, split + ".lmdb")
        is_train = split == "train"
        dataset = LMDBDataset(
            split_path,
            gzip=self.args.gzip,
            sample_cluster=self.args.sample_cluster,
        )
        # normalize target if needed
        if is_train:
            if self.args.mol_target_key is not None and self.args.mol_target_normalize:
                mol_mean, mol_std = dataset.calc_mean_std(
                    self.args.mol_target_key, self.args.mol_target_idx
                )
                self.stats["mol_mean"] = mol_mean
                self.stats["mol_std"] = mol_std
            else:
                self.stats["mol_mean"] = 0.0
                self.stats["mol_std"] = 1.0
            # normalize target if needed
            if (
                self.args.atom_target_key is not None
                and self.args.atom_target_normalize
            ):
                atom_mean, atom_std = dataset.calc_mean_std(
                    self.args.atom_target_key, self.atom_target_idx
                )
                self.stats["atom_mean"] = atom_mean
                self.stats["atom_std"] = atom_std
            else:
                self.stats["atom_mean"] = 0.0
                self.stats["atom_std"] = 1.0

        dataset = DataTypeDataset(dataset, self.args.data_type)
        atom_type_key = self.args.atom_type_key
        atom_pos_key = self.args.atom_pos_key
        cluster_path = None
        if self.args.sample_cluster:
            cluster_path = (
                self.args.train_cluster_path
                if is_train
                else self.args.valid_cluster_path
            )
            assert os.path.exists(cluster_path), f"File {cluster_path} not exists!"
            conf_dataset = SampleClusterDataset(
                dataset,
                self.seed,
                cluster_path,
            )
        elif is_train or not self.args.expand_valid_dataset:
            conf_dataset = ConformationSampleDataset(
                dataset,
                self.seed,
                atom_pos_key,
            )
        else:
            repeat_count = self.args.repeat_count
            conf_dataset = ConformationExpandDataset(
                dataset,
                self.seed,
                atom_pos_key,
                repeat_count=repeat_count,
            )

        sample_dataset = GridDataset(
            conf_dataset,
            self.seed,
            self.dictionary,
            atom_type_key,
            atom_pos_key,
            is_train,
            self.args,
        )

        nest_dataset = NestedDictionaryDataset(
            {
                "batched_data": sample_dataset,
            },
        )
        if is_train:
            nest_dataset = EpochShuffleDataset(
                nest_dataset, len(nest_dataset), self.seed
            )

        print("| Loaded {} with {} samples".format(split, len(nest_dataset)))
        self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        return model

    def disable_shuffling(self) -> bool:
        return False

    def train_step(self, sample, model, loss, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        sample.update(self.stats)
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, loss, test=False):
        model.eval()
        sample.update(self.stats)
        with torch.no_grad():
            loss, sample_size, logging_output = loss(model, sample)
        return loss, sample_size, logging_output
