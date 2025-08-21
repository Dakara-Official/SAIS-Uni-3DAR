#!/bin/bash
#SBATCH --job-name=uni3dar                      # 作业名称
#SBATCH --nodes=1                              # 请求节点数
#SBATCH --ntasks-per-node=1                    # 每个节点任务数
#SBATCH --gpus-per-node=4                      # 每个节点GPU数
#SBATCH --cpus-per-task=24                      # 每个任务CPU核心数
#SBATCH --mem=50G                              # 每个节点内存
#SBATCH --partition=gpu                        # 分区名称
#SBATCH --time=3-00:00:00                      # 最大运行时间
#SBATCH --output=logs/%j.out                   # 输出日志路径
#SBATCH --error=logs/%j.err                    # 错误日志路径
#SBATCH -w node13
# # 设置GPU使用相关环境变量
# export CUDA_VISIBLE_DEVICES=3  # !!! 重要：指定只使用第1张GPU !!!
# export LOCAL_RANK=0            # 本地GPU排名设置为0

# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_USE_CUDA_DSA=1

# 从Slurm环境变量中获取配置
[ -z "${MASTER_PORT}" ] && MASTER_PORT=10089
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)  # 获取主节点地址

export MASTER_ADDR
export MASTER_PORT
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# 设置OpenMP环境变量
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=$WORLD_SIZE
[ -z "${OMPI_COMM_WORLD_RANK}" ] && OMPI_COMM_WORLD_RANK=$RANK

# 计算GPU数量
n_gpu=$(nvidia-smi -L | wc -l)
echo "Total GPUs on node: $n_gpu"
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)

# 设置默认参数
[ -z "${data_type}" ] && data_type=molecule_energy
[ -z "${lr}" ] && lr=3e-4
[ -z "${min_lr}" ] && min_lr=1e-9
[ -z "${warmup_steps}" ] && warmup_steps=30000
[ -z "${total_steps}" ] && total_steps=500000
[ -z "${update_freq}" ] && update_freq=1
[ -z "${seed}" ] && seed=1
[ -z "${clip_norm}" ] && clip_norm=1
[ -z "${weight_decay}" ] && weight_decay=1e-4
[ -z "${merge_level}" ] && merge_level=6
[ -z "${layer}" ] && layer=12
[ -z "${batch_size}" ] && batch_size=16
[ -z "${emb_dim}" ] && emb_dim=768
[ -z "${head_num}" ] && head_num=12

# 解析命令行参数
data_path=$1
[ -z "${more_args}" ] && more_args=""

echo "more_args" $more_args

# 设置输出路径
[ -z "${base_dir}" ] && base_dir=./results
base_name=$2
save_dir=$base_dir/$base_name
[ -z "${wandb_project}" ] && wandb_project=your_wandb_project

# 创建目录
tmp_save_dir=./tmp_ckpt
mkdir -p $tmp_save_dir
mkdir -p $save_dir
cat $(pwd)/$0 > ${save_dir}/save_orders
printenv > ${save_dir}/environment_variables

# 设置日志文件路径（使用Slurm分配的RANK）
log_save_dir=${save_dir}/log_${RANK}.txt

# 设置PyTorch和OMP参数
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
echo "n_gpu per node" $n_gpu
echo "WORLD_SIZE" $WORLD_SIZE
echo "RANK" $RANK
echo "LOCAL_RANK" $LOCAL_RANK
echo "MASTER_ADDR" $MASTER_ADDR
echo "MASTER_PORT" $MASTER_PORT


# 设置Wandb参数
export WANDB_DISABLED=true
export WANDB_MODE=offline
# export WANDB_API_KEY=xxxxxxx

set -o pipefail

torchrun --nproc_per_node=$n_gpu \
        $(which unicore-train) $data_path --user-dir ./uni3dar --train-subset train --valid-subset valid \
        --num-workers 8 --ddp-backend=no_c10d \
        --task uni3dar --loss ar --arch uni3dar \
        --bf16 --tensorboard-logdir $save_dir/tsb \
        --emb-dim $emb_dim --num-head $head_num  \
        --layer $layer \
        --wandb-project $wandb_project --wandb-name $base_name \
        --log-interval 100 --log-format simple \
        --save-interval-updates 10000 --validate-interval-updates 10000 --keep-interval-updates 2 --no-epoch-checkpoints  \
        --save-dir $save_dir/ckpt --tmp-save-dir $tmp_save_dir \
        --batch-size $batch_size \
        --data-buffer-size 32 --fixed-validation-seed 11 --batch-size-valid $((batch_size * 2)) \
        --optimizer adam --adam-betas '(0.9, 0.99)' --adam-eps 1e-6 --clip-norm $clip_norm \
        --lr $lr --update-freq $update_freq \
        --weight-decay $weight_decay \
        --seed $seed \
        --data-type $data_type --merge-level $merge_level  \
        --warmup-updates $warmup_steps --max-update $total_steps \
        --ema-decay 0.999 --validate-with-ema \
        --lr-scheduler cosine --warmup-init-lr 1e-9 --min-lr $min_lr \
        --grid-len 0.24  --gzip --H-prob 1.0 --xyz-resolution 0.01 --recycle 1  \
        --loss-ratio-tree 1.0 --loss-ratio-atom 1.0 --loss-ratio-xyz 0.1 \
        --tree-delete-start-layer 1 --tree-delete-ratio 0.1 \
        --atom-type-key atom_type --atom-pos-key atom_pos --allow-atoms H,C,N,O,F,P,S,Cl,Br  --head-dropout 0.1 \
        --energy-min-stat -4.88198375701904 --energy-max-stat 124.046897888183 --energy-padding 0.05 --energy-condition-noise 0.05 \
        $more_args \
        2>&1 | tee -a ${log_save_dir}

exit $?