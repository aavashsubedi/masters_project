#!/bin/bash
#SBATCH --job-name=CombinatoricsRL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --constraint=A100
#SBATCH --exclusive

echo "Starting run at: `date`"

while true; do
    source /share/nas2/lislaam/masters_project/venv/bin/activate
    echo "Activated virtual environment"
    wandb login
    export WANDB_API_KEY= 8e782a594dad15c64868ccff129984a8a344af28
    python /share/nas2/lislaam/masters_project/src/rl_pipeline/run_experiment.py
done
