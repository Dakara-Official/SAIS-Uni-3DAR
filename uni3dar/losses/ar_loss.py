from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss
import numpy as np
from .metric_reduce import (
    reduce_regression,
    reduce_bi_classification,
    reduce_regression_atom,
    reduce_bi_classification_atom,
)
import torch.nn as nn


@register_loss("ar")
class ARLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.args = task.args
        self.softmax = nn.Softmax(dim=-1)

    def gen_loss(self, gen_outputs, loss, sample_size, logging_output):
        (
            pred_list,
            gt_list,
            name_list,
            count_list,
            ratio_list,
            gen_sample_size,
        ) = gen_outputs
        for i in range(len(pred_list)):
            name = name_list[i]
            preds = pred_list[i]
            gts = gt_list[i]
            cur_loss = 0
            cur_count = 0
            preds = [preds] if not isinstance(preds, list) else preds
            gts = [gts] if not isinstance(gts, list) else gts
            for j in range(len(preds)):
                if len(preds[j].shape) == 0:
                    # diffusion loss
                    cur_loss += preds[i]
                else:
                    cur_loss += F.nll_loss(
                        F.log_softmax(preds[j], dim=-1, dtype=torch.float32),
                        gts[j].long(),
                        reduction="none",
                    )
                cur_count += count_list[i]
            loss += cur_loss.sum() * ratio_list[i]
            logging_output[name + "_metric"] = cur_loss.sum().data
            logging_output[name + "_count"] = cur_count

        sample_size += gen_sample_size
        return loss, sample_size, logging_output

    def pred_loss(
        self,
        pred_outputs,
        atom_mean,
        atom_std,
        mol_mean,
        mol_std,
        loss,
        sample_size,
        logging_output,
    ):
        (
            pred_list,
            gt_list,
            name_list,
            count_list,
            ratio_list,
            keys,
            pred_sample_size,
        ) = pred_outputs

        def process_reduce(
            name,
            preds,
            gts,
            mean,
            std,
            keys,
            is_atom_task,
            is_classification,
            logging_output,
        ):
            if is_atom_task and self.args.atom_num_classes > 2:
                preds = preds.reshape(-1, 3)
                gts = gts.reshape(-1, 3)
            else:
                preds = preds.reshape(-1)
                gts = gts.reshape(-1)
            if not is_classification:
                preds = (preds * std) + mean
            else:
                preds = self.softmax(
                    preds.view(-1, self.args.atom_num_classes).float()
                )[:, 1]
                gts = gts.long()
            name = name + "_reg" if not is_classification else name + "_cls"
            if is_atom_task:
                logging_output[name + "_pred"] = preds.cpu().numpy()
                logging_output[name + "_label"] = gts.cpu().numpy()
                logging_output[name + "_atom_key"] = keys[0].cpu().numpy()
                logging_output[name + "_mol_key"] = keys[1].cpu().numpy()
            else:
                logging_output[name + "_pred"] = preds.cpu().numpy()
                logging_output[name + "_label"] = gts.cpu().numpy()
                logging_output[name + "_mol_key"] = keys.cpu().numpy()
            return logging_output

        for i in range(len(pred_list)):
            name = name_list[i]
            preds = pred_list[i]
            gts = gt_list[i]
            cur_loss = 0
            is_atom_task = "atom_target" in name
            (mean, std) = (atom_mean, atom_std) if is_atom_task else (mol_mean, mol_std)
            is_classification = False
            is_gt_multidim = len(gts.shape) > 1 and gts.shape[1] > 1
            if is_atom_task:
                is_classification = (
                    self.args.atom_num_classes > 1 and not is_gt_multidim
                )
            else:
                is_classification = self.args.mol_num_classes > 1 and not is_gt_multidim

            if is_classification:
                cur_loss += F.nll_loss(
                    F.log_softmax(preds, dim=-1, dtype=torch.float32),
                    gts.long(),
                    reduction="none",
                )
            else:
                cur_gt = gts.float()
                cur_gt = (cur_gt - mean) / std
                cur_loss += F.l1_loss(
                    preds.reshape(-1),
                    cur_gt.reshape(-1),
                    reduction="none",
                )

            loss += cur_loss.sum() * ratio_list[i]
            logging_output[name + "_metric"] = cur_loss.sum().data
            logging_output[name + "_count"] = count_list[i]
            if not self.training:
                logging_output = process_reduce(
                    name,
                    preds,
                    gts,
                    mean,
                    std,
                    keys[i],
                    is_atom_task,
                    is_classification,
                    logging_output,
                )

        sample_size += pred_sample_size
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        batched_data = sample["batched_data"]
        (
            gen_outputs,
            preds_outputs,
            dec_lens,
            data_process_time,
        ) = model(batched_data)

        bsz = dec_lens.shape[0] - 1

        loss = 0.0
        sample_size = 0
        logging_output = {}

        loss, sample_size, logging_output = self.gen_loss(
            gen_outputs, loss, sample_size, logging_output
        )
        loss, sample_size, logging_output = self.pred_loss(
            preds_outputs,
            sample["atom_mean"],
            sample["atom_std"],
            sample["mol_mean"],
            sample["mol_std"],
            loss,
            sample_size,
            logging_output,
        )

        logging_output["loss"] = loss.data
        logging_output["sample_size"] = sample_size
        logging_output["grid_size_sum"] = dec_lens[-1].item()
        logging_output["data_time_sum"] = data_process_time.sum().item()
        logging_output["raw_atom_count"] = sum(batched_data["raw_atom_count"])
        logging_output["cutoff_atom_count"] = sum(batched_data["cutoff_atom_count"])
        logging_output["cutoff_ratio"] = (
            np.array(batched_data["cutoff_atom_count"]) > 0
        ).sum()

        logging_output["bsz"] = bsz

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        """Aggregate logging outputs from data parallel training."""

        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sum_grid_size = sum(
            float(log.get("grid_size_sum", 0)) for log in logging_outputs
        )
        sum_data_time = sum(
            float(log.get("data_time_sum", 0)) for log in logging_outputs
        )
        sum_cutoff_atom_count = sum(
            float(log.get("cutoff_atom_count", 0)) for log in logging_outputs
        )
        sum_cutoff_ratio = sum(
            float(log.get("cutoff_ratio", 0)) for log in logging_outputs
        )
        sum_raw_atom_count = sum(
            float(log.get("raw_atom_count", 0)) for log in logging_outputs
        )

        avg_grid_size = float(sum_grid_size / bsz)
        avg_data_time = float(sum_data_time / bsz)
        avg_cutoff_atom_count = float(sum_cutoff_atom_count / bsz)
        avg_cutoff_ratio = float(sum_cutoff_ratio / bsz)
        avg_raw_atom_count = float(sum_raw_atom_count / bsz)
        metrics.log_scalar("avg_grid_size", avg_grid_size, bsz, round=2)
        metrics.log_scalar("avg_data_time", avg_data_time, bsz, round=6)
        metrics.log_scalar("cutoff_atom_count", avg_cutoff_atom_count, bsz, round=6)
        metrics.log_scalar("cutoff_ratio", avg_cutoff_ratio, bsz, round=6)
        metrics.log_scalar("raw_atom_count", avg_raw_atom_count, bsz, round=6)

        for key in logging_outputs[0].keys():
            if ("loss" in key and "ft" not in key) or "_metric" in key:
                total_loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                cur_size = sample_size
                count_key = key.replace("_metric", "_count")
                if "_metric" in key and count_key in logging_outputs[0]:
                    cur_size = sum(log.get(count_key, 0) for log in logging_outputs)
                else:
                    count_key = "loss_count"
                metrics.log_scalar(key, total_loss_sum / cur_size, cur_size, round=8)
                metrics.log_scalar(count_key, cur_size / bsz, bsz, round=8)

        if "train" not in split:
            keys = logging_outputs[0].keys()
            reduce_names = []
            for key in keys:
                if "_pred" in key:
                    reduce_names.append(key[:-5])
            for name in reduce_names:
                is_atom = "atom" in name
                is_cls = "_cls" in name
                pred_key = name + "_pred"
                label_key = name + "_label"
                atom_key = name + "_atom_key"
                mol_key = name + "_mol_key"
                preds = np.concatenate(
                    [log.get(pred_key, 0) for log in logging_outputs],
                    axis=0,
                )
                gts = np.concatenate(
                    [log.get(label_key, 0) for log in logging_outputs], axis=0
                )
                keys = np.concatenate(
                    [log.get(mol_key, 0) for log in logging_outputs], axis=0
                )
                if is_atom:
                    atom_keys = np.concatenate(
                        [log.get(atom_key, 0) for log in logging_outputs], axis=0
                    )
                    keys = (keys, atom_keys)
                if is_atom:
                    reduce_func = (
                        reduce_bi_classification_atom
                        if is_cls
                        else reduce_regression_atom
                    )
                else:
                    reduce_func = (
                        reduce_bi_classification if is_cls else reduce_regression
                    )
                res = reduce_func(preds, gts, keys)
                for key in res:
                    metrics.log_scalar(
                        f"{split}_{name}_{key}",
                        res[key],
                        1,
                        round=8,
                    )

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return is_train
