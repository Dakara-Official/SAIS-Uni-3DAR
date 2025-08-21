#!/usr/bin/env python3 -u
# Copyright (c) DP Technology.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import logging
import sys
import time
import os, io, json
import re
from typing import Optional, Any, Callable
from data import LMDBDataset

import numpy as np
import torch
from unicore import (
    options,
    tasks,
    utils,
)
from unicore.distributed import utils as distributed_utils
from tqdm import tqdm
# from data.crystal_data_utils import match_rate_at_k 不需要
import pickle # 输出成pkl文件的库


def check_files_count(filename, total_count=10000, data_type="molecule"):
    try:
        if data_type == "molecule":
            molecule_count = 0
            with open(filename, "r") as file:
                while True:
                    line = file.readline()
                    if not line:
                        break
                    try:
                        atom_count = int(line.strip())
                        molecule_count += 1
                        for _ in range(atom_count + 1):
                            file.readline()
                    except ValueError:
                        continue
            if molecule_count == total_count:
                return True
        elif data_type == "crystal":

            with open(filename, "r") as f:
                content = f.read()
            blocks = re.split(r"(?=^[ \t]*data_)", content, flags=re.MULTILINE)
            if "data_" not in blocks[0]:
                blocks = blocks[1:]
            if len(blocks) == total_count:
                return True
    except Exception as e:
        print(e)

    return False


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("unicore_cli.train")

# 以下几个函数都不需要
# def inference_molecule(model, total_n, output_file):
#     if check_files_count(output_file, total_count=total_n, data_type="molecule"):
#         return
#     try:
#         os.system(f"rm {output_file}")
#     except:
#         pass
#     count = 0
#     while count < total_n:
#         res, _ = model.generate()
#         if len(res) > 0:
#             with open(f"{output_file}", "a+") as f:
#                 for cur_res in res:
#                     if count % 1000 == 0:
#                         print(f"count: {count}")
#                     if count < total_n:
#                         f.write(cur_res)
#                         f.write("\n")
#                     count += 1


# def inference_crystal(model, total_n, output_file):
#     from ase import Atoms
#     from ase.io import write

#     if check_files_count(output_file, total_count=total_n, data_type="crystal"):
#         return
#     try:
#         os.system(f"rm {output_file}")
#     except:
#         pass
#     count = 0
#     while count < total_n:
#         res, _ = model.generate()
#         if len(res) > 0:
#             with open(f"{output_file}", "a+") as f:
#                 for cur_res in res:
#                     if count % 1000 == 0:
#                         print(f"count: {count}")
#                     if count < total_n:
#                         cif_buffer = io.BytesIO()
#                         write(cif_buffer, cur_res, format="cif")
#                         cur_res = cif_buffer.getvalue().decode("utf-8")
#                         f.write(cur_res)
#                         f.write("\n")
#                     count += 1


# def inference_crystal_cond(args, dataset, model, total_n, output_file):
#     from ase import Atoms

#     world_size = distributed_utils.get_data_parallel_world_size()
#     rank = distributed_utils.get_data_parallel_rank()

#     res_name = os.path.split(output_file)[-1]
#     score_dict = {
#         total_n: {
#             "match": 0,
#             "total": 0,
#             "rmse_sum": 0.0,
#         },
#     }
#     shuffle_idx = np.arange(len(dataset))
#     np.random.shuffle(shuffle_idx)
#     for inner_i in tqdm(range(len(shuffle_idx))):
#         index = shuffle_idx[inner_i]
#         if index % world_size != rank:
#             continue
#         cur_data = dataset[index]
#         gt_structure = Atoms(
#             symbols=cur_data[args.atom_type_key],
#             cell=cur_data[args.lattice_matrix_key],
#             scaled_positions=np.array(cur_data[args.atom_pos_key]).reshape(-1, 3),
#             pbc=True,
#         )
#         gt_atoms = np.array(sorted(gt_structure.get_atomic_numbers())).reshape(-1) - 1
#         cur_res = []
#         cur_scores = []
#         try_cnt = 0
#         max_try = 10
#         min_generated_samples = total_n * 20
#         count = 0
#         while count < min_generated_samples:
#             res, score = model.generate(data=cur_data, atom_constraint=gt_atoms)
#             for i in range(len(res)):
#                 cur_atoms = (
#                     np.array(sorted(res[i].get_atomic_numbers())).reshape(-1) - 1
#                 )
#                 if (gt_atoms.shape[0] == cur_atoms.shape[0]) and np.all(
#                     cur_atoms == gt_atoms
#                 ):
#                     cur_res.append(res[i])
#                     cur_scores.append(score[i])
#                     count += 1
#                 if count >= min_generated_samples:
#                     break
#             try_cnt += 1
#             if try_cnt > max_try:
#                 break
#             atom_match_rate = count / (len(res) + 1e-5)
#             if atom_match_rate <= 0.1 and try_cnt > 2:
#                 break
#         sorted_idx = np.argsort(cur_scores)
#         cur_res = [cur_res[i] for i in sorted_idx]
#         for eval_key in score_dict:
#             match, rmse = match_rate_at_k(gt_structure, cur_res[:eval_key], eval_key)
#             score_dict[eval_key]["match"] += match
#             score_dict[eval_key]["total"] += 1
#             score_dict[eval_key]["rmse_sum"] += rmse
#             cur_cnt = score_dict[eval_key]["total"]
#             cur_match = score_dict[eval_key]["match"] / (cur_cnt + 1e-12)
#             cur_rmse = score_dict[eval_key]["rmse_sum"] / (
#                 score_dict[eval_key]["match"] + 1e-12
#             )
#             print(
#                 f"{res_name}-r{rank}-c{cur_cnt}, Top-{eval_key}, mr: {cur_match}, rmse: {cur_rmse}"
#             )

#         with open(
#             f"{output_file}_r{rank}_bs{args.batch_size}.json",
#             "w",
#         ) as f:
#             json.dump(score_dict, f, indent=2)

#     total_processed_samples = 0
#     while total_processed_samples < len(shuffle_idx):
#         all_res = {
#             total_n: {
#                 "match": 0,
#                 "total": 0,
#                 "rmse_sum": 0.0,
#             },
#         }
#         for i in range(world_size):
#             cur_json_file = f"{output_file}_r{i}_bs{args.batch_size}.json"
#             if os.path.exists(cur_json_file):
#                 with open(cur_json_file, "r") as f:
#                     cur_res = json.load(f)
#                     for eval_key in score_dict:
#                         all_res[eval_key]["match"] += cur_res[str(eval_key)]["match"]
#                         all_res[eval_key]["total"] += cur_res[str(eval_key)]["total"]
#                         all_res[eval_key]["rmse_sum"] += cur_res[str(eval_key)][
#                             "rmse_sum"
#                         ]
#         total_processed_samples = all_res[total_n]["total"]
#         time.sleep(5)

#     if rank == 0:
#         for eval_key in all_res:
#             cur_cnt = all_res[eval_key]["total"]
#             cur_match = all_res[eval_key]["match"] / (cur_cnt)
#             cur_rmse = all_res[eval_key]["rmse_sum"] / (all_res[eval_key]["match"])
#             print(
#                 f"{res_name}-r{rank}-c{cur_cnt} agg, Top-{eval_key}, mr: {cur_match}, rmse: {cur_rmse}"
#             )
#         with open(
#             f"{output_file}_bs{args.batch_size}.json",
#             "w",
#         ) as f:
#             json.dump(all_res, f, indent=2)

# 以下三个函数为完全新建的函数
# 能量归一化函数
def normalize_energy(raw_energy, min_val, max_val, padding=0.05):
    data_range = max_val - min_val
    buffer = padding * data_range
    adjusted_min = min_val - buffer
    adjusted_max = max_val + buffer
    
    # 边界保护
    raw_energy = max(min(raw_energy, adjusted_max), adjusted_min)
    
    # 应用归一化
    normalized = (raw_energy - adjusted_min) / (adjusted_max - adjusted_min)
        
    # 确保值在合理范围
    return np.clip(normalized, 0.0, 1.0)

def inference_molecule_energy(args, model, condition_file):
    """
    基于能量条件生成分子
    """
    import pandas as pd
    
    # 读取能量条件文件
    try:
        # 1. 路径处理 + 2. 编码处理 + 3. 分隔符处理
        df = pd.read_csv(
            condition_file,
            encoding='utf-8',        # 备选: 'ISO-8859-1', 'gbk'
            sep=r',\s*',             # 处理逗号后的空白符
            engine='python'
        )
        
        # 4. 列名清洗
        df.columns = df.columns.str.strip().str.replace('\ufeff', '')
        
        # 5. 验证列存在性
        if "Value" not in df.columns:
            raise KeyError(f"'Value'列不存在，实际列名: {df.columns.tolist()}")
        
        energy_values = df["Value"].values
        idx_values = df["idx"].values

        logger.info(f"成功加载 {len(energy_values)} 条数据")

        
    except Exception as e:
        logger.error(f"文件读取失败: {str(e)}")
        # 详细诊断
        if not os.path.exists(condition_file):
            logger.error(f"路径不存在: {condition_file}")
        elif os.path.isfile(condition_file):
            with open(condition_file, 'r') as f:
                sample = f.readline()
                logger.error(f"文件首行示例: {repr(sample)}")


    # 批处理逻辑
    batch_size = args.batch_size
    parsed_molecules = []
        
    # 按批次处理能量值
    for i in range(0, len(energy_values), batch_size):
        batch_end = min(i + batch_size, len(energy_values))
        batch_energies = energy_values[i:batch_end]
        batch_indices = idx_values[i:batch_end]
            
        # 归一化批处理能量值
        norm_energies = [
            normalize_energy(energy, args.energy_min_stat, args.energy_max_stat, args.energy_padding)
            for energy in batch_energies
        ]
            
        # 构造能量数据字典
        energy_data = {"properties": norm_energies}
            
        # 批处理生成分子
        molecules, scores = model.generate(data=energy_data)
            
        # 处理生成的分子
        for j, cur_res in enumerate(molecules):
            mol_data = parse_xyz_string(cur_res)
            if mol_data:
                # 添加对应的idx值
                mol_data["idx"] = int(batch_indices[j])
                parsed_molecules.append(mol_data)
  
    with open("/saisresult/submit.pkl", "wb") as f:
        pickle.dump(parsed_molecules, f)
    print(parsed_molecules)
    logger.info(f"Total generated molecules: {len(energy_values)}")

def parse_xyz_string(xyz_str):
    """解析内存中的XYZ格式字符串为结构化数据"""
    lines = xyz_str.strip().split('\n')
    if not lines:
        return None
    
    # 解析原子总数和注释行
    natoms = int(lines[0].strip())  # 第一行转为整数
    comment = lines[1] if len(lines) > 1 else ""  # 第二行注释（可选）
    
    # 解析原子坐标
    elements = []
    coordinates = []
    for i in range(2, 2 + natoms):  # 从第三行开始是原子数据
        if i >= len(lines):
            break
        parts = lines[i].split()
        if len(parts) < 4:  # 确保有元素+XYZ坐标
            continue
        elements.append(parts[0])
        try:
            # 将坐标转为浮点数
            coords = list(map(float, parts[1:4]))
            coordinates.append(coords)
        except ValueError:
            continue  # 忽略格式错误的行
    
    return {
        "natoms": natoms,
        "elements": elements,
        "coordinates": coordinates
    }


def main(args) -> None:
    utils.import_user_module(args)
    utils.set_jit_fusion_options()

    assert (
        args.batch_size is not None
    ), "Must specify batch size either with --batch-size"

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    args.model = "uni3dar_sampler"
    # Print args
    logger.info(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    assert args.loss, "Please specify loss to train a model"

    # Build model and loss
    model = task.build_model(args)
    state = torch.load(args.finetune_from_model, map_location="cpu", weights_only=False)
    errors = model.load_state_dict(state["ema"]["params"], strict=True)
    print("loaded from {}, errors: {}".format(args.finetune_from_model, errors))

    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(getattr(p, "_orig_size", p).numel() for p in model.parameters()),
            sum(
                getattr(p, "_orig_size", p).numel()
                for p in model.parameters()
                if p.requires_grad
            ),
        )
    )

    # 不需要
    # total_n = args.num_samples
    # output_file = args.save_path

    model = model.cuda().bfloat16()
    start = time.time()
    # 不需要
    # if args.data_type == "molecule":
    #     inference_molecule(model, total_n, output_file)
    # elif args.data_type == "crystal":
    #     if args.crystal_pxrd > 0 or args.crystal_component > 0:
    #         dataset = LMDBDataset(
    #             os.path.join(args.data, "test.lmdb"),
    #             key_to_id=True,
    #             gzip=args.gzip,
    #             sample_cluster=False,
    #         )
    #         inference_crystal_cond(
    #             args,
    #             dataset,
    #             model,
    #             total_n,
    #             output_file,
    #         )
    #     else:
    #         inference_crystal(model, total_n, output_file)
    if args.data_type == "molecule_energy":
        inference_molecule_energy(args, model, args.condition_file) # 新增条件生成
    end = time.time()
    print(f"Total time: {end - start}")


def cli_main(
    modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None,
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)
    try:
        distributed_utils.call_main(args, main)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
            time.sleep(1)
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    cli_main()
