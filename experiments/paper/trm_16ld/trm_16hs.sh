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


# Activate conda environment
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# Make paths
cd $SLURM_SUBMIT_DIR

SCRIPT_DIR="$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SCRIPT_DIR/tdmpc2/train.py"

SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=True \
        use_task_embedding=True \
        obs="state" \
        model_size="S" \
        mlp_t=True \
        halt_max_steps=0 \
        halt_exploration_prob=0 \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="dmc_trmd_s_s${SEED}_16hs16ld_16opt_no_rec" \
        enable_wandb=True \
        H_cycles=1 \
        L_cycles=1 \
        num_state_obs_per_token=16 \
        pooling_strategy="mean" \
        hidden_size=16 \
        num_heads=2 \
        seed="$SEED" \
        latent_dim=16 &
done

wait
