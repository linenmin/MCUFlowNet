#!/bin/bash -l
#SBATCH --job-name=deployFT_mainline
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=45G
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

# Mainline (EdgeFlowNet transpose-conv). Differences from v3 slurms:
#  - arch_family edgeflownet_mainline (no arch_code)
#  - init from EdgeFlowNet/checkpoints/best.ckpt via explicit_path
#  - flow_divisor 1.0 (mainline training never divided GT)
python wrappers/run_deploy_ft_one.py \
  --config configs/retrain_v3_deploy_ft.yaml \
  --experiment_name retrain_v3_deploy_ft_run1 \
  --model_name mainline \
  --arch_family edgeflownet_mainline \
  --init_mode explicit_path \
  --init_ckpt_path EdgeFlowNet/checkpoints/best.ckpt \
  --flow_divisor 1.0 \
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
  --prefetch_batches 4 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --gpu_device 0 \
  --seed 42
