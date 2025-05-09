#!/bin/bash
#SBATCH --job-name=train            # Job name
#SBATCH --time=3:00:00             # Maximum runtime (hh:mm:ss)
#SBATCH --ntasks=1                  # Number of tasks (usually 1 for a simple job)
#SBATCH --output=/dev/null          # Output file
#SBATCH --error=/dev/null           # Error file
#SBATCH --cpus-per-task=1           # Number of CPUs per task
#SBATCH --mem=32GB                  # Memory allocation
#SBATCH --partition=long-grace      # Partition name (change to match your cluster)
#SBATCH --gres=gpu:1                # Request 1 GPU (if needed)

# Activate virtual environment (if needed)
source venv/bin/activate

# Run your script
python train.py --wandb_id loyilpo0
