#!/bin/bash -l
#SBATCH --job-name=retrainV3_ft3d
#SBATCH --clusters=genius
#SBATCH --account=lp_embaivision
#SBATCH --partition=gpu_p100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=27
#SBATCH --mem=135G
#SBATCH --gpus-per-node=3
#SBATCH --time=72:00:00
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

python wrappers/run_retrain_v3_batch.py \
  --stage ft3d \
  --config configs/retrain_v3_ft3d.yaml \
  --candidates_csv plan/retrain_v3/retrain_v3_candidates.csv \
  --experiment_name retrain_v3_ft3d_run1 \
  --fc2_experiment_dir outputs/retrain_v3_fc2/retrain_v3_fc2_run1 \
  --init_ckpt_name best \
  --gpu_devices 0,1,2 \
  --max_workers 3 \
  --num_epochs 50 \
  --batch_size 32 \
  --micro_batch_size 32 \
  --ft3d_num_workers 6 \
  --ft3d_eval_num_workers 2 \
  --prefetch_batches 2 \
  --eval_prefetch_batches 0 \
  --ft3d_eval_every_epoch 1 \
  --sintel_eval_every_epoch 2 \
  --grad_clip_global_norm 200.0
