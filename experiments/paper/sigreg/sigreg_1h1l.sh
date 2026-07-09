#!/bin/bash

#SBATCH --job-name=newt_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=71:59:00
#SBATCH --gres=gpu:1


# If using Slurm, run this script from the root (.../newt) directory of the repository!
#
# SIGReg regularization (no SimNorm) across all three recursive dynamics architectures
# (TRM / SimpleTRM / SRM) with no recursion (1 H-cycle x 1 L-cycle), as a non-recursive
# baseline. No skip connections, no FiLM. Three runs share one GPU.


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
cd $SLURM_SUBMIT_DIR

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SCRIPT_DIR/tdmpc2/train.py"

SEEDS=(0)

# TRM dynamics
for SEED in "${SEEDS[@]}"; do
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics="trm" \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_trm_s${SEED}_16ld_1h1l_nofilm_noskip" \
        enable_wandb=True \
        H_cycles=1 \
        L_cycles=1 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=False \
        wm_regularization_type="sigreg" &
done

# SimpleTRM dynamics
for SEED in "${SEEDS[@]}"; do
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics="simple" \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_simple_s${SEED}_16ld_1h1l_nofilm_noskip" \
        enable_wandb=True \
        H_cycles=1 \
        L_cycles=1 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=False \
        use_simple_trm_skip_connections=False \
        wm_regularization_type="sigreg" &
done

# SRM dynamics
for SEED in "${SEEDS[@]}"; do
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics="srm" \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_srm_s${SEED}_16ld_1h1l_nofilm_noskip" \
        enable_wandb=True \
        H_cycles=1 \
        L_cycles=1 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=False \
        srm_truncation_length=1 \
        wm_regularization_type="sigreg" &
done

wait
