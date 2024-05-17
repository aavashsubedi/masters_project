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
source /share/nas2/lislaam/masters_project/.venv/bin/activate
echo "Activated virtual environment"
#for i in {1..5}; do
#while true; do
 #   pass
python /share/nas2/lislaam/masters_project/src/rl_pipeline/run_experiment.py
#done
