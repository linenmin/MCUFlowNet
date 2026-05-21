#!/bin/bash -l
#SBATCH --job-name=hybrid_v2_group_b_opus
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gpus-per-node=4
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Lem1780474486@outlook.com

# search_hybrid_v2 ablation: group (b) = NSGA-II + Warmstart only
# 模型: Anthropic Claude Opus 4.7 (anthropic/claude-opus-4-7)
# Prompt: v2 (TASK SEMANTICS / OPTIMIZATION OBJECTIVE / 极简 warmstart)
# 单 LLM call (gen 0 之前调一次), 预算 < $0.5
# 预计墙钟: ~3-3.5h (主要是 800 个 arch 的 NPU 评估; warmstart 调用约 30s)

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [ -z "${ANTHROPIC_API_KEY}" ]; then
  echo "ERROR: ANTHROPIC_API_KEY not set." >&2
  echo "Add: export ANTHROPIC_API_KEY='sk-ant-...' to ~/.bashrc and re-login." >&2
  exit 1
fi

python wrappers/run_nsga2_search.py \
  --config configs/nsga2_v4.yaml \
  --experiment_name "group_b_v4_opus" \
  --output_root "outputs/search_hybrid_v2" \
  --ablation_group b \
  --supernet_experiment_dir outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill \
  --gpu_devices 0,1,2,3 \
  --max_workers 4 \
  --num_workers 4 \
  --prefetch_batches 2
