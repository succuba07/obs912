#!/bin/bash
#SBATCH --job-name=oracle_training       # 作业名称
#SBATCH --partition=xahdtest            # 指定分区
#SBATCH --nodes=1                       # 申请1个计算节点
#SBATCH --ntasks-per-node=1             # 每个节点运行1个任务
#SBATCH --gres=dcu:1                    # 申请1块DCU卡
#SBATCH --output=%j.out                 # 标准输出日志（%j是作业ID）
#SBATCH --error=%j.err                  # 标准错误日志
#SBATCH --time=24:00:00                  # 设置最大运行时间

# 加载必要的环境模块，例如Python、CUDA等
module load anaconda3
conda activate tf_env

# 执行您的训练脚本
python ./predicts/train.py
