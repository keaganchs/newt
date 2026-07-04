#!/bin/bash

# Activate conda environment
# export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

eval "$(/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/bin/conda shell.bash hook)"
conda activate newt

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")")")"
PYTHON_SCRIPT="${PROJECT_ROOT}/tdmpc2/train.py"
cd "${PROJECT_ROOT}"
echo "Using Python script: ${PYTHON_SCRIPT}"

SEEDS=(0 1 2)

# ---- latent_dim=16, 1h1l (no recursion) ----
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
        wandb_run_name="smp_s${SEED}_16ld_1h1l_simnorm" \
        wandb_group="paper_simple_16ld_1h1l" \
        enable_wandb=True \
        H_cycles=1 \
        L_cycles=1 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=False \
        use_simple_trm_skip_connections=False \
        rrm_mask_x_for_y_update=True \
        wm_regularization_type="simnorm" &
done

# ---- latent_dim=16, 4h3l (H_cycles=4, L_cycles=3) ----
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
        wandb_run_name="smp_s${SEED}_16ld_4h3l_simnorm" \
        wandb_group="paper_simple_16ld_4h3l" \
        enable_wandb=True \
        H_cycles=4 \
        L_cycles=3 \
        seed="$SEED" \
        latent_dim=16 \
        use_film_dynamics=False \
        use_simple_trm_skip_connections=False \
        rrm_mask_x_for_y_update=True \
        wm_regularization_type="simnorm" &
done

wait
