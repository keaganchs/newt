#!/bin/bash
#SBATCH --job-name=smp_notask
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=71:59:00
#SBATCH --gres=gpu:1

# Task-embedding ablation (control): the LEGACY paper_simple_384ld_4h3l config
# with use_task_embedding=False (task_dim=0: the dynamics input is [z | a] and
# task identity only reaches the model through the observation). Carry init is
# the default (y copies the encoded WM latent); the companion _randinit script
# crosses this with rrm_random_y_init=True.
#
# L_layers=1 reproduces the legacy single-layer core (the paper_simple runs
# predate the L_layers refactor; their stored L_layers=2 was unused).
#
# 3090 cluster (SBATCH). Submit from the repo root:  sbatch <this file>
# 1 cell x 3 seeds share one GPU.
# group: paper_simple_384ld_4h3l_notask

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

cd "$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SLURM_SUBMIT_DIR/tdmpc2/train.py"
SEEDS=(0 1 2)
GROUP="paper_simple_384ld_4h3l_notask"

for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task=dmcontrol \
        num_envs=21 \
        obs=state \
        model_size=S \
        use_trm_encoder=False \
        use_task_embedding=False \
        use_trm_dynamics=simple \
        latent_dim=384 \
        hidden_size=16 \
        H_cycles=4 \
        L_cycles=3 \
        L_layers=1 \
        srm_truncation_length=3 \
        rrm_random_y_init=False \
        use_film_dynamics=False \
        wm_regularization_type=simnorm \
        xl_dynamics_mlp=False \
        use_simple_trm_skip_connections=False \
        rrm_mask_x_for_y_update=True \
        compile=True \
        log_trm_gradnorms=True \
        seed="$SEED" \
        enable_wandb=True \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_group="${GROUP}" \
        wandb_run_name="smp_s${SEED}_384ld_4h3l_notask" &
done

wait
