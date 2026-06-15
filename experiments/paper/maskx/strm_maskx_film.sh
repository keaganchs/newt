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
        use_trm_dynamics="simple" \
        use_task_embedding=True \
        obs="state" \
        model_size="S" \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_run_name="smp_s${SEED}_16ld_2h6l_sisw_maskx_film" \
        enable_wandb=True \
        H_cycles=2 \
        L_cycles=6 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=True \
        use_simple_trm_skip_connections=True \
        simple_trm_skip_type="swiglu" \
        rrm_mask_x_for_y_update=True &
done

wait
