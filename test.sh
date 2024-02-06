#!/bin/bash
#SBATCH --job-name=CombLearning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --constraint=A100
#SBATCH --exclusive
#SBATCH --nodelist=compute-0-9

echo "Starting run at: `date`"
while true; do
    pass
done
