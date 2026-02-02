#!/bin/bash

#SBATCH --job-name=newt_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=stud
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6000
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1


# If using Slurm, run this script from the root (.../newt) directory of the repository!


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
cd $SLURM_SUBMIT_DIR

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SCRIPT_DIR/tdmpc2/train.py"

# Run
python3 $PYTHON_SCRIPT \
    task="mujoco" \
    use_trm_encoder=True \
    use_task_embedding=False \
    obs="state" \
    model_size="S" \
    mlp_t=True \
    halt_max_steps=0 \
    wandb_project="newt_trm" \
    wandb_entity="keagan" \
    wandb_run_name="trm_mlp_s_0" \
    enable_wandb=True \
