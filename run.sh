#!/bin/bash
#SBATCH --job-name=latent_control
#SBATCH --output=/network/scratch/l/leo.gagnon/sbatch_output.txt
#SBATCH --error=/network/scratch/l/leo.gagnon/sbatch_error.txt
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=unkillable

# Activate modules and load venv
source ~/latent_control/venv/bin/activate
python ~/latent_control/train.py