#!/bin/bash
#SBATCH --job-name=CombinatorialRL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --constraint=A100
#SBATCH --exclusive

nvidia-smi 
echo ">>>start"
source /lislaam/masters_project/venv/bin/activate
echo ">>>training"
python /lislaam/masters_project/src/rl_pipeline/run_experiment.py