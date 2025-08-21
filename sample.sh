#!/bin/bash
#SBATCH --job-name=uni3dar                      # 作业名称
#SBATCH --nodes=1                              # 请求节点数
#SBATCH --ntasks-per-node=1                    # 每个节点任务数
#SBATCH --gpus-per-node=1                     # 每个节点GPU数
#SBATCH --cpus-per-task=24                      # 每个任务CPU核心数
#SBATCH --mem=50G                              # 每个节点内存
#SBATCH --partition=gpu                        # 分区名称
#SBATCH --time=3-00:00:00                      # 最大运行时间
#SBATCH --output=logs/%j.out                   # 输出日志路径
#SBATCH --error=logs/%j.err                    # 错误日志路径
#SBATCH -w node13

# 在Slurm中运行时，使用单节点单任务（单GPU）配置
[ -z "${MASTER_PORT}" ] && MASTER_PORT=10088
[ -z "${MASTER_IP}" ] && MASTER_IP=127.0.0.1
# 强制设置为单GPU运行
n_gpu=1

echo "Total GPUs on node: $n_gpu"
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

[ -z "${seed}" ] && seed=1
[ -z "${merge_level}" ] && merge_level=6

[ -z "${data_path}" ] && data_path=None
[ -z "${layer}" ] && layer=12
[ -z "${batch_size}" ] && batch_size=10
[ -z "${emb_dim}" ] && emb_dim=768
[ -z "${head_num}" ] && head_num=12

[ -z "${more_args}" ] && more_args=""

[ -z "${tree_temperature}" ] && tree_temperature=0.9
[ -z "${atom_temperature}" ] && atom_temperature=0.3
[ -z "${xyz_temperature}" ] && xyz_temperature=0.3
[ -z "${count_temperature}" ] && count_temperature=1.0
[ -z "${rank_by}" ] && rank_by="atom"
[ -z "${data_type}" ] && data_type=molecule_energy
if [ $data_type == "molecule_energy" ]; then
    lastname=xyz
fi
[ -z "${save_path}" ] && save_path=$1_res_s${seed}_tt${tree_temperature}_at${atom_temperature}_xt${xyz_temperature}_ct${count_temperature}_ns${num_samples}_rr${rank_ratio}_rb${rank_by}

echo "save_path" $save_path

# 设置日志文件路径（使用Slurm分配的RANK）
log_save_dir=${save_dir}/log.txt


# 使用CUDA_VISIBLE_DEVICES指定要使用的GPU
export CUDA_VISIBLE_DEVICES=0  # 只使用第一张GPU

torchrun uni3dar/inference.py $data_path --user-dir uni3dar --train-subset train --valid-subset valid \
    --num-workers 8 --ddp-backend=c10d \
    --task uni3dar --loss ar --arch uni3dar_sampler \
    --bf16 \
    --emb-dim $emb_dim --num-head $head_num  \
    --layer $layer \
    --batch-size $batch_size\
    --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $((batch_size * 2)) \
    --seed $seed \
    --data-type $data_type --merge-level $merge_level  \
    --grid-len 0.24  --xyz-resolution 0.01 --recycle 1  \
    --tree-temperature $tree_temperature --atom-temperature $atom_temperature --xyz-temperature $xyz_temperature --count-temperature $count_temperature \
    --rank-by $rank_by \
    --finetune-from-model $1 \
    --allow-atoms H,C,N,O,F,P,S,Cl,Br \
    --energy-min-stat -4.88198375701904 --energy-max-stat 124.046897888183 --energy-padding 0.05 --energy-condition-noise 0.05 \
    --condition-file /saisdata/input_condition.csv \
    $more_args