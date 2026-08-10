#!/bin/bash
#SBATCH --job-name=abd-reorient
#SBATCH -t 24:00:00
#SBATCH -c 2
#SBATCH --mem-per-cpu=16G
#SBATCH -p stud
#SBATCH --gres=gpu:1
#SBATCH --array=1-5
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

# Go to the directory from which sbatch was called.
cd "$SLURM_SUBMIT_DIR"
mkdir -p slurm_logs cluster_runs

SEED=$SLURM_ARRAY_TASK_ID

echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SEED"
echo "Working directory: $(pwd)"
nvidia-smi

# Activate the same environment you use locally.
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_GPU_ALLOCATOR=cuda_malloc_async

source .venv/bin/activate

python -u train_cluster.py \
  --env_name LeapCubeReorient \
  --policy_type abd \
  --num_timesteps 80000000 \
  --num_envs 8192 \
  --num_evals 8 \
  --learning_rate 0.0001 \
  --hidden_dim 128 \
  --encoder_dim 512 256 \
  --decoder_dim 256 \
  --activation relu \
  --seed "$SEED" \
  --output_dir cluster_runs
