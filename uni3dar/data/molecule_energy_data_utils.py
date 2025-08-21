# 以下仿照源代码crystal_data_utils.py构建的molecule_energy_data_utils.py
import numpy as np
import torch

def get_molecule_energy_cond(
    args, cond_data, is_train, dictionary, xyz_null_id, bsz=1
):
    assert args.data_type == "molecule_energy"
    level = args.merge_level + 1

    def normalize_energy(raw_energy, min_val, max_val, padding=0.05):
        """分子能量归一化函数
        
        Args:
            raw_energy: 原始能量值
            min_val: 数据集最小值
            max_val: 数据集最大值
            padding: 边界扩展比例 (默认5%)
        
        Returns:
            归一化后的值，范围[0,1]
        """
        # 计算缓冲量
        data_range = max_val - min_val
        buffer = padding * data_range
        
        # 调整边界
        adjusted_min = min_val - buffer
        adjusted_max = max_val + buffer
        
        # 应用归一化
        normalized = (raw_energy - adjusted_min) / (adjusted_max - adjusted_min)
        
        # 确保值在合理范围
        return np.clip(normalized, 0.0, 1.0)

    if args.data_type == "molecule_energy":
        raw_energy = cond_data["properties"]
        energy_batches = []
        for _ in range(bsz):
            # 应用归一化
            energy_norm = normalize_energy(
                raw_energy,
                min_val=args.energy_min_stat,
                max_val=args.energy_max_stat,
                padding=args.energy_padding
            )

            # 训练时的特殊处理
            if is_train:
                # 添加可控噪声
                if args.energy_condition_noise > 0:
                    noise = np.random.normal(0, args.energy_condition_noise)
                    energy_norm += noise

                # 随机条件丢弃
                if np.random.rand() < args.molecule_energy_cond_drop:
                    energy_norm = 0.0

            energy_batches.append(energy_norm)
        energy_arr = np.stack(energy_batches, axis=0)

        energy_arr = energy_arr.reshape(-1, 1)  # 添加特征维度

    tokens = []

    def add_token(token, token_list):
        token = np.array(token, dtype=np.int32)
        token_list.append(token)

    if args.data_type == "molecule_energy":
        add_token(
            [dictionary["[ENERGY]"]],
            tokens,
        )

    tokens = np.concatenate(tokens, axis=0) if tokens else np.zeros(0, dtype=np.int32)
    feat = {}
    feat["decoder_type"] = tokens
    feat["decoder_level"] = np.full(tokens.shape[0], level, dtype=np.int32)
    feat["decoder_xyz"] = np.full((tokens.shape[0], 3), xyz_null_id, dtype=np.int32)
    half_grid_size = 2 ** (args.merge_level) * 0.5 * args.grid_len
    feat["decoder_phy_pos"] = np.full((tokens.shape[0], 3), half_grid_size)
    feat["decoder_is_second_atom"] = np.full(tokens.shape[0], False, dtype=np.bool_)
    feat["decoder_remaining_atoms"] = np.full(tokens.shape[0], 0, dtype=np.int32)
    feat["decoder_remaining_tokens"] = np.full(tokens.shape[0], 0, dtype=np.int32)
    feat["decoder_count"] = np.full(tokens.shape[0], 0, dtype=np.int32)

    if bsz > 1:
        for key in feat:
            feat[key] = feat[key][None].repeat(bsz, axis=0)

    feat["energy"] = energy_arr.astype(np.float32)
    return feat