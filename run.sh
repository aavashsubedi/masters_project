#!/bin/bash
#SBATCH --job-name=CombLearning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
#SBATCH --constraint=A100
#SBATCH --exclusive

echo "Starting run at: `date`"
source /share/nas2/asubedi/masters_project/venv/bin/activate
echo "Activated virtual environment"

python /share/nas2/asubedi/masters_project/main.py