#!/bin/bash

#SBATCH --job-name=newt_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=2000
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1


# If using Slurm, run this script from the root of the repository!


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/../tdmpc2/train.py"

# Run
python3 $PYTHON_SCRIPT \
    task="mujoco" \
    use_trm_encoder=True \
    obs="state" \
    model_size="S" \
    mlp_t=True \
    wandb_project="newt_trm" \
    wandb_entity="keagan" \
    wandb_run_name="trm_mlp_s_0" \
    enable_wandb=True \