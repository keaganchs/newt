#!/bin/bash
#SBATCH --job-name=ld16_1h1l
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=71:59:00
#SBATCH --gres=gpu:1

# latent_dim sweep, SimpleTRM @1h1l, latent_dim=16 (low-recursion).
#
# 3090 cluster (SBATCH). Submit from the repo root:  sbatch <this file>
# Low-recursion only (3-day wall limit); 1 cell x 3 seeds share one GPU.
# group: abl_latent_dim_1h1l/smp_16ld_1h1l  (3 seeds aggregate; filter by "abl_latent_dim_1h1l/" prefix for the plan)

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

cd "$SLURM_SUBMIT_DIR"
PYTHON_SCRIPT="$SLURM_SUBMIT_DIR/tdmpc2/train.py"
SEEDS=(0 1 2)
RUN="smp_16ld_1h1l"
GROUP="abl_latent_dim_1h1l"

for SEED in "${SEEDS[@]}"; do
    python3 "$PYTHON_SCRIPT" \
        task=dmcontrol \
        num_envs=21 \
        obs=state \
        model_size=S \
        use_trm_encoder=False \
        use_task_embedding=True \
        use_trm_dynamics=simple \
        latent_dim=16 \
        hidden_size=16 \
        H_cycles=1 \
        L_cycles=1 \
        use_film_dynamics=False \
        wm_regularization_type=simnorm \
        use_dis_loss=False \
        xl_dynamics_mlp=False \
        use_simple_trm_skip_connections=False \
        rrm_mask_x_for_y_update=True \
        compile=True \
        log_trm_gradnorms=False \
        seed="$SEED" \
        enable_wandb=True \
        wandb_project="TRM Dynamics" \
        wandb_entity="trm-dynamics" \
        wandb_group="${GROUP}/${RUN}" \
        wandb_run_name="${RUN}_s${SEED}" &
done

wait
