#!/bin/bash -l
#SBATCH --job-name=ablV1_a4_gate4x_only
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=45000M
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Lem1780474486@outlook.com

# Ablation V1 new variant: A4 = edgeflownet_bilinear_gate4x
# Bilinear upsampling + 1/4 global broadcast gate, NO bottleneck ECA.
# Purpose: separate the contribution of the gate from the contribution of ECA
# in the bilinear+ECA+gate4x backbone. Ordering becomes:
#   A1 bilinear (no ECA, no gate)
#   A2 bilinear + ECA            (ECA only)
#   A4 bilinear + gate4x         (gate only)              <-- this run
#   A3 bilinear + ECA + gate4x   (full)
# The skeleton matches Supernet V3 (gate uses the encoder bottleneck as
# context, applied to the 1/4 feature feeding OutLow). See
# plan/ablation_v1/findings.md (2026-05-09 section).
#
# Shares experiment_name with the A2/A3 skeleton-aligned rerun so all
# corrected runs land under the same parent prefix in
# outputs/ablation_v1_fc2/.

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python wrappers/run_ablation_v1_fc2.py \
  --config configs/ablation_v1_fc2.yaml \
  --experiment_name ablation_v1_fc2_run2_p100_skeleton_aligned \
  --variants edgeflownet_bilinear_gate4x \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-4 \
  --lr_min 1.0e-6 \
  --grad_clip_global_norm 200.0 \
  --fc2_num_workers 8 \
  --fc2_eval_num_workers 4
