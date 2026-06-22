#!/bin/bash

#SBATCH --job-name=newt_trm
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=amd2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=71:59:00
#SBATCH --gres=gpu:1


# If using Slurm, run this script from the root (.../newt) directory of the repository!
#
# SIGReg variants of the Newt default-dynamics baselines (no TRM/SimpleTRM/SRM): the default
# S model (384ld), the same with a 16-dim latent, and the same with the XL dynamics MLP
# ([512, 512] hidden dims) toggled via the xl_dynamics_mlp flag. All use SIGReg regularization
# (wm_regularization_type="sigreg") instead of SimNorm. Three runs share one GPU.


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
cd "$SLURM_SUBMIT_DIR"
mkdir -p log

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SCRIPT_DIR/tdmpc2/train.py"

SEEDS=(0)

# Newt default S (384ld)
for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_dmc_newt_s_s${SEED}" \
        enable_wandb=True \
        seed="$SEED" \
        latent_dim=384 \
        wm_regularization_type="sigreg" &
done

# Newt S with 16ld
for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_dmc_newt_s_s${SEED}_16ld" \
        enable_wandb=True \
        seed="$SEED" \
        latent_dim=16 \
        wm_regularization_type="sigreg" &
done

# Newt S with the XL dynamics MLP ([512, 512] hidden dims)
for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="sigreg_dmc_newt_s_s${SEED}_xl_dynamics_model" \
        enable_wandb=True \
        seed="$SEED" \
        latent_dim=384 \
        xl_dynamics_mlp=True \
        wm_regularization_type="sigreg" &
done

wait
