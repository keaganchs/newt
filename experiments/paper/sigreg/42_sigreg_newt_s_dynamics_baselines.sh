#!/bin/bash

# Determined cluster launcher: SIGReg variants of the Newt default-dynamics baselines
# (no TRM/SimpleTRM/SRM). Runs 3 seeds for each of the 3 experiments concurrently (9 runs):
#   1. Newt default S (384ld)
#   2. Newt S with a 16-dim latent
#   3. Newt S with the XL dynamics MLP ([512, 512] hidden dims, via xl_dynamics_mlp)
# All use SIGReg regularization (wm_regularization_type="sigreg") instead of SimNorm.

# Activate conda environment
# export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

eval "$(/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/bin/conda shell.bash hook)"
conda activate newt

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")")"
PYTHON_SCRIPT="${PROJECT_ROOT}/tdmpc2/train.py"
cd "${PROJECT_ROOT}"
echo "Using Python script: ${PYTHON_SCRIPT}"

# SEEDS=(0 1 2) # 9 runs concurrently
SEEDS=(0) # 3 runs concurrently

# Newt default S (384ld)
for SEED in "${SEEDS[@]}"; do
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        use_task_embedding=True \
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
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        use_task_embedding=True \
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
    python3 $PYTHON_SCRIPT \
        task="dmcontrol" \
        num_envs=21 \
        use_trm_encoder=False \
        use_trm_dynamics=None \
        use_task_embedding=True \
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
