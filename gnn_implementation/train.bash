#!/bin/bash
# This is train.bash
#SBATCH --job-name=training-job       # Name of your job
#SBATCH --mem-per-cpu=16G
#SBATCH --output=slurm_logs/result_%A_%a.log        # File where standard output goes (%j inserts Job ID)
#SBATCH --partition=stud            # Request the GPU partition
#SBATCH --nodes=1                    # Request exactly 1 physical node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Give that task 4 CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=24:00:00              # Set a hard limit of 24 hours
#SBATCH --array=1

echo "This will run on the compute node!"

SEED=${SLURM_ARRAY_TASK_ID}
echo "Running seed ${SEED}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
python3 --version
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# per-seed logdir to avoid collisions
BASE_LOGDIR="/home/stud_nguyen1/gnn_implementation/logs/LeapCubeReorient-abd"
LOGDIR="${BASE_LOGDIR}-seed_${SEED}"
mkdir -p "${LOGDIR}"

echo "Starting training abd (seed ${SEED})..."
python train_jax_ppo.py \
    --env_name=LeapCubeReorient \
    --use_tb=True \
    --logdir="${LOGDIR}" \
    --num_videos=1 \
    --num_timesteps=180000000 \
    --num_envs=8192 \
    --num_evals=20 \
    --learning_rate=0.0001 \
    --use_wandb=False \
    --policy_type=abd \
    --seed=${SEED}

echo "Training completed!"


