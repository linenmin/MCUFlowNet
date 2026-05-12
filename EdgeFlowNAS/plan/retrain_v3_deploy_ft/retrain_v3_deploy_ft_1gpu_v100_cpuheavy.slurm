#!/bin/bash -l
#SBATCH --job-name=deployFT_v3_v100cpu
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=Lem1780474486@outlook.com

cd $VSC_DATA/test/MCUFlowNet/EdgeFlowNAS/

module load TensorFlow/2.15.1-foss-2023a-CUDA-12.1.1
source ~/tf_work/bin/activate

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Why this exists vs the plain 1-V100 slurm:
# The plain 1-V100 slurm gave the data pipeline only 4 cpu cores +
# ft3d_num_workers=2. At input 157x203 the V100 finishes a fwd+bwd
# batch in ~50ms but the dataloader needs ~4s to prepare a batch from
# FT3D PNG+PFM files. Result: GPU starves, per-epoch wall time blew
# up to ~90 min (observed on first v100 run, epoch 1=5187s, epoch
# 2=5381s; 20 epochs × 4 candidates would exceed the 24h cap by ~5x).
#
# Fix: ask for 16 cores on the V100, bump ft3d_num_workers to 8 and
# prefetch_batches to 6. This restores per-GPU-core ratio comparable
# to the 3xP100 retrain_v3 (which used 9 cores/GPU). Mem still 80G —
# 16-worker concurrent PFM decode peaks well under that.
python wrappers/run_deploy_ft_batch.py \
  --config configs/retrain_v3_deploy_ft.yaml \
  --candidates_csv plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv \
  --experiment_name retrain_v3_deploy_ft_run1_v100cpu \
  --gpu_devices 0 \
  --max_workers 1 \
  --num_epochs 20 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-6 \
  --lr_schedule constant \
  --early_stop_patience 5 \
  --grad_clip_global_norm 50.0 \
  --input_height 157 \
  --input_width 203 \
  --ft3d_num_workers 8 \
  --ft3d_eval_num_workers 2 \
  --prefetch_batches 6 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --seed 42
