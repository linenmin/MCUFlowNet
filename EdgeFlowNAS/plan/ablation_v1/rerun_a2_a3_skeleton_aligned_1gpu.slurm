#!/bin/bash -l
#SBATCH --job-name=ablV1_rerun_a2a3
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

# Ablation V1 rerun: A2 (edgeflownet_bilinear_eca) + A3 (edgeflownet_bilinear_eca_gate4x)
# Reason: previous run1 used a mis-aligned skeleton -- ECA was placed after the
# decoder stack instead of at the encoder bottleneck, and the 1/4 global gate
# was applied to the 1/2 head input. This does not match Supernet V3
# (efnas/network/MultiScaleResNet_supernet_v3.py). Corrected skeleton now puts
# ECA at the encoder bottleneck and gates the 1/4 feature feeding OutLow.
# See plan/ablation_v1/findings.md (2026-05-09 section).
#
# Both variants are trained sequentially in this single job so all rerun
# outputs land under one parent experiment directory. A new experiment_name
# preserves the legacy run1 outputs under outputs/ablation_v1_fc2/.

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python wrappers/run_ablation_v1_fc2.py \
  --config configs/ablation_v1_fc2.yaml \
  --experiment_name ablation_v1_fc2_run2_p100_skeleton_aligned \
  --variants edgeflownet_bilinear_eca,edgeflownet_bilinear_eca_gate4x \
  --gpu_device 0 \
  --num_epochs 400 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-4 \
  --lr_min 1.0e-6 \
  --grad_clip_global_norm 200.0 \
  --fc2_num_workers 8 \
  --fc2_eval_num_workers 4
