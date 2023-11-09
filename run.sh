#!/bin/bash
#SBATCH --job-name=CombLearning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --constraints=A100
#SBATCH --time=24:00:00
#SBATCH --exclusive

# export WANDB_CACHE_DIR=./wandb_cache
python main.py