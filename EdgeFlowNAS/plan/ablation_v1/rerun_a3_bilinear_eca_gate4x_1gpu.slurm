#!/bin/bash -l
#SBATCH --job-name=ablV1_rerun_a3_gate4x
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=45G
#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Lem1780474486@outlook.com

# Ablation V1 rerun: A3 = edgeflownet_bilinear_eca_gate4x
# Reason: previous run1 placed ECA after the decoder stack and applied the
# 1/4 global broadcast gate to the 1/2 head input, both inconsistent with
# Supernet V3 (efnas/network/MultiScaleResNet_supernet_v3.py). Corrected
# skeleton now puts ECA at the encoder bottleneck and gates the 1/4 feature
# feeding OutLow. See plan/ablation_v1/findings.md (2026-05-09 section).
# Writes to a new experiment name so run1 outputs are preserved.

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python wrappers/run_ablation_v1_fc2.py \
  --config configs/ablation_v1_fc2.yaml \
  --experiment_name ablation_v1_fc2_run2_p100_skeleton_aligned \
  --variants edgeflownet_bilinear_eca_gate4x \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-4 \
  --lr_min 1.0e-6 \
  --grad_clip_global_norm 200.0 \
  --fc2_num_workers 8 \
  --fc2_eval_num_workers 4
