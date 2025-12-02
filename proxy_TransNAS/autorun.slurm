#!/bin/bash
#SBATCH --job-name=autorun_proxyEvaluate
#SBATCH --clusters=wice
#SBATCH --partition=gpu_a100
#SBATCH --account=intro_vsc37996        
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00                  # 最长跑 8 小时，超时自动杀
#SBATCH --output=logs/zico_%j.out        # 标准输出 & 报错保存位置

# 1. 进入你的项目目录
cd /vsc-hard-mounts/leuven-data/379/vsc37996/test/MCUFlowNet/proxy_TransNAS

# 2. 加载必要模块（如果你平时需要 module）
# module load CUDA/xx.xx 之类的，看你环境（如果你用的是纯 conda 就可以不写）

# 3. 激活你的 conda 环境
eval "$(${VSC_DATA}/miniconda3/bin/conda shell.bash hook)"
conda activate nas

# 4. （可选）打印一下环境信息
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
nvidia-smi

# 5. 执行你的长任务（这里举例）
python full_run_transnas.py \
  --proxies zico myscore \
  --search_spaces micro macro \
  --total_samples 200 \
  --chunk_size 100 \
  --batch_size 16

echo "Job finished at $(date)"
