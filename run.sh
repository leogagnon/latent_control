#!/bin/bash
#SBATCH --job-name=latent_control
#SBATCH --output=/network/scratch/l/leo.gagnon/sbatch_output.txt
#SBATCH --error=/network/scratch/l/leo.gagnon/sbatch_error.txt
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --cpus-per-task=4

# Activate modules and load venv
source ~/latent_control/venv/bin/activate
python ~/latent_control/notebooks/runfig.py