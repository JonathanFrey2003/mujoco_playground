#!/bin/bash
# This is train.bash
#SBATCH --job-name=training-job       # Name of your job
#SBATCH --partition=stud            # Request the GPU partition
#SBATCH --output=slurm_logs/result_%j.log  
#SBATCH --nodes=1                    # Request exactly 1 physical node
#SBATCH --ntasks=1                   # Run a single task
#SBATCH --cpus-per-task=4            # Give that task 4 CPU cores
#SBATCH --gres=gpu:1                 # Request 1 GPU
#SBATCH --time=24:00:00              # Set a hard limit of 2 hours

echo "This will run on the compute node!"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
python3 --version
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
#pip install --upgrade pip
#pip install -r requirements.txt

echo "Starting inference..."
#echo "Test if model can adapt to mass shifting during inference"
echo "Inference for GNN, version 20260807-165231"
python train_jax_ppo.py \
    --env_name=LeapCubeReorient \
    --use_tb=True \
    --load_checkpoint_path=/home/stud_nguyen1/gnn_implementation/logs/LeapCubeReorient-gnn/LeapCubeReorient-20260807-165231/checkpoints \
    --logdir=/home/stud_nguyen1/gnn_implementation/logs/rollout \
    --num_videos=2 \
    --num_timesteps=0 \
    --policy_type=gnn \
    --play_only=True \
    --use_wandb=False \
    --use_tb=False &

# echo "Inference for MLP baseline, version 20260802-165851"
# python train_jax_ppo.py \
#     --env_name=LeapCubeReorient \
#     --use_tb=True \
#     --load_checkpoint_path=/home/stud_nguyen1/gnn_implementation/logs/LeapCubeReorient-gnn/LeapCubeReorient-20260801-214838/checkpoints \
#     --logdir=/home/stud_nguyen1/gnn_implementation/logs/rollout \
#     --num_videos=2 \
#     --num_timesteps=0 \
#     --policy_type=mlp \
#     --play_only=True \
#     --use_wandb=False \
#     --use_tb=False &

wait


echo "Inference completed!"

