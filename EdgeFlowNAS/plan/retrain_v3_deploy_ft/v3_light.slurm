#!/bin/bash -l
#SBATCH --job-name=deployFT_v3_light
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=45000M
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

python wrappers/run_deploy_ft_one.py \
  --config configs/retrain_v3_deploy_ft.yaml \
  --experiment_name retrain_v3_deploy_ft_run1 \
  --model_name v3_light \
  --arch_family fixed_v3 \
  --arch_code "0,0,0,0,0,0,0,0,0,0,0" \
  --init_mode experiment_dir \
  --init_experiment_dir outputs/retrain_v3_ft3d/retrain_v3_ft3d_run1 \
  --init_ckpt_name sintel_best \
  --flow_divisor 12.5 \
  --num_epochs 20 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --lr 1.0e-5 \
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
