#!/bin/bash -l
#SBATCH --job-name=deployFT_v3
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=180G
#SBATCH --gpus-per-node=4
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

# 4 GPUs, 4 candidates → max_workers=4 (one candidate per GPU, in parallel).
# Sintel eval runs every 2 epochs as a diagnostic; ckpt selection uses
# FT3D val EPE (per plan/retrain_v3_deploy_ft/findings.md §9).
python wrappers/run_deploy_ft_batch.py \
  --config configs/retrain_v3_deploy_ft.yaml \
  --candidates_csv plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv \
  --experiment_name retrain_v3_deploy_ft_run1 \
  --gpu_devices 0,1,2,3 \
  --max_workers 4 \
  --num_epochs 20 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-6 \
  --lr_schedule constant \
  --early_stop_patience 5 \
  --grad_clip_global_norm 50.0 \
  --input_height 157 \
  --input_width 203 \
  --ft3d_num_workers 6 \
  --ft3d_eval_num_workers 2 \
  --prefetch_batches 2 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --seed 42
