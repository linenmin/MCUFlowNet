#!/bin/bash -l
#SBATCH --job-name=hybrid_v1_group_b
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Lem1780474486@outlook.com

# Phase 5 ablation: group (b) = NSGA-II + Warmstart
# Expected wall clock: ~3h (single LLM warmstart call + 16 NSGA-II generations)

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# IMPORTANT: GEMINI_API_KEY 必须在提交前已 export 到 shell 或写在 ~/.bashrc 里
# slurm 会继承. 检查:
if [ -z "${GEMINI_API_KEY}" ]; then
  echo "ERROR: GEMINI_API_KEY not set. Add 'export GEMINI_API_KEY=...' to ~/.bashrc and re-login." >&2
  exit 1
fi

python wrappers/run_nsga2_search.py \
  --config configs/nsga2_v3.yaml \
  --experiment_name "group_b" \
  --output_root "outputs/ablation_phase5" \
  --ablation_group b \
  --supernet_experiment_dir outputs/supernet/edgeflownas_supernet_v3_fc2_172x224_run1_archparallel_distill \
  --gpu_devices 0,1,2,3 \
  --max_workers 4 \
  --num_workers 4 \
  --prefetch_batches 2
