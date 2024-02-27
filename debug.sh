#!/bin/bash
#SBATCH --job-name=Combinatorial_RL
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --constraint=A100
#SBATCH --exclusive
#SBATCH --nodelist=compute-0-0

#echo "Starting run at: `date`"

while true; do
    source /share/nas2/lislaam/masters_project/.venv/bin/activate
    python /share/nas2/lislaam/masters_project/src/rl_pipeline/run_experiment.py
done