#!/bin/bash -l
#SBATCH --job-name=deployFT_v3_1gpu
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
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

# Single V100 fallback: 4 candidates run SEQUENTIALLY on one GPU
# (max_workers=1). Per-candidate wall time on V100 ≈ 2× faster than P100
# (~2.5h estimated), so 4 candidates × ~2.5h ≈ 10h, well within the
# 24h cap. dataloader workers reduced to fit 4 CPU cores.
python wrappers/run_deploy_ft_batch.py \
  --config configs/retrain_v3_deploy_ft.yaml \
  --candidates_csv plan/retrain_v3_deploy_ft/retrain_v3_deploy_ft_candidates.csv \
  --experiment_name retrain_v3_deploy_ft_run1_v100 \
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
  --ft3d_num_workers 2 \
  --ft3d_eval_num_workers 1 \
  --prefetch_batches 2 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --seed 42
