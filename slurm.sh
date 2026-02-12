#!/bin/bash
#SBATCH --job-name=NC_training    # Job name
#SBATCH --partition=ENSTA-h100        # Partition/queue name (depends on your cluster)
#SBATCH --nodes=1                  # Number of tasks (usually 1 for Python scripts)
#SBATCH --cpus-per-task=4           # Number of CPU cores
#SBATCH --gres=gpu:1                  
#SBATCH --time=04:00:00             # Max runtime (HH:MM:SS)
#SBATCH --output=logs/%x_%j.out     # Standard output (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err      # Standard error

# Activate venv if needed
source source .venv/bin/activate

# Run your Python script
python3 -m src.train