#!/bin/bash

#SBATCH --job-name=newt_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=amd2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1


# If using Slurm, run this script from the root (.../newt) directory of the repository!


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
cd "$SLURM_SUBMIT_DIR"
mkdir -p log

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SCRIPT_DIR/tdmpc2/train.py"

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=False \
        obs="state" \
        model_size="M" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="dmc_newt_m_s${SEED}" \
        enable_wandb=True \
        seed="$SEED" \
        latent_dim=128 &
done

wait
