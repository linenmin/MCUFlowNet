#!/bin/bash -l
#SBATCH --job-name=retrainV3_acc_ft3d
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=43G
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
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

python wrappers/run_retrain_v3_ft3d.py \
  --config configs/retrain_v3_ft3d.yaml \
  --experiment_name retrain_v3_ft3d_run1 \
  --model_name v3_acc \
  --arch_code 0,1,2,2,2,2,0,0,0,0,1 \
  --resume \
  --resume_experiment_name retrain_v3_ft3d_run1 \
  --resume_ckpt_name last \
  --gpu_device 0 \
  --num_epochs 50 \
  --batch_size 32 \
  --micro_batch_size 16 \
  --ft3d_num_workers 4 \
  --ft3d_eval_num_workers 1 \
  --prefetch_batches 1 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --grad_clip_global_norm 200.0
