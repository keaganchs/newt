#!/bin/bash
# ---------------------------------------------------------------------------
# Shared setup + launcher for the TRM-dynamics ablation batches (42 style).
#
# Every ablation batch script `source`s this file, then calls `launch` once per
# config cell. `launch` runs the cell for all 3 seeds in the background (the 42
# convention: pack many runs on one GPU with `&`, then `wait`).
#
# The single source of truth for the ANCHOR config lives in `launch` below. Each
# experiment is a *slice at the anchor*: a batch overrides only the axis it
# sweeps and inherits everything else, so an "off"/"baseline" cell is byte-for-
# byte the anchor. This is the same launch path, env, hydra keys, seed handling
# and run naming as the hand-written 42 scripts (e.g. 42/latent_dim/384ld.sh);
# it is just centralised so the ~90 config cells stay consistent and DRY.
#
# ANCHOR (held fixed unless a batch overrides it):
#   use_trm_dynamics=simple  latent_dim=384  hidden_size=384
#   use_film_dynamics=False   wm_regularization_type=simnorm
#   use_dis_loss=False        xl_dynamics_mlp=False
#   use_simple_trm_skip_connections=False
#   rrm_mask_x_for_y_update=True
#   compile=True              log_trm_gradnorms=False
#   H_cycles=4  L_cycles=3    (the S-preset "recursion on" depth; the plan fixes
#                              everything except cycles, so 4h3l is the anchor
#                              depth for experiments that do not sweep cycles)
#
# NOTE on hidden_size: the SimpleTRM (anchor) MLP path uses latent_dim internally
# and ignores hidden_size; we still set hidden_size=latent_dim so the trm/srm
# transformer variants (which DO use it) scale with the latent under the same
# knob. num_heads is left at the S preset (=2), which evenly divides every
# latent_dim we sweep (16/128/384/512), and simnorm_dim=8 divides them too.
# ---------------------------------------------------------------------------

# NOTE: intentionally no `set -u` here -- `conda activate` references unset vars
# and would abort under it (the hand-written 42 scripts omit it for the same reason).

# ---- environment (identical to experiments/paper/42/*.sh) -----------------
# export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=disable
export LD_LIBRARY_PATH="/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

eval "$(/pfss/mlde/workspaces/mlde_wsp_PI_Deramo/miniconda3/bin/conda shell.bash hook)"
conda activate newt

# ---- paths ----------------------------------------------------------------
# This file lives at <newt>/experiments/paper/42/ablation/_ablation_common.sh,
# so the repo root is five directories up from a sourcing batch script. Resolve
# from THIS file so it is correct no matter who sources it.
_ABL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_ABL_DIR}/../../../.." && pwd)"
PYTHON_SCRIPT="${PROJECT_ROOT}/tdmpc2/train.py"
cd "${PROJECT_ROOT}"
echo "[ablation] repo root: ${PROJECT_ROOT}"
echo "[ablation] train.py : ${PYTHON_SCRIPT}"

# ---- seeds ----------------------------------------------------------------
SEEDS=(0 1 2)

# ---- launcher -------------------------------------------------------------
# usage: launch <run_name> <wandb_group> [key=val ...]
#   <run_name>    encodes axis + key hyperparameters; "_s<SEED>" is appended.
#   <wandb_group> the unique group for this SET of runs (for pulling plot data).
#   key=val ...   hydra overrides that move this cell off the anchor.
# Overrides are merged into the anchor BY KEY (so we never emit a duplicate
# hydra key, which hydra rejects). One backgrounded run is launched per seed.
launch() {
    local run_name="$1"; shift
    local wandb_group="$1"; shift

    # -- anchor (single source of truth) --
    declare -A C=(
        [task]="dmcontrol"
        [num_envs]="21"
        [obs]="state"
        [model_size]="S"
        [use_trm_encoder]="False"
        [use_task_embedding]="True"
        [use_trm_dynamics]="simple"
        [latent_dim]="384"
        [hidden_size]="384"
        [use_film_dynamics]="False"
        [wm_regularization_type]="simnorm"
        [use_dis_loss]="False"
        [xl_dynamics_mlp]="False"
        [use_simple_trm_skip_connections]="False"
        [rrm_mask_x_for_y_update]="True"
        [compile]="True"
        [log_trm_gradnorms]="False"
        [H_cycles]="4"
        [L_cycles]="3"
        [enable_wandb]="True"
        [wandb_project]="TRM Dynamics"
        [wandb_entity]="trm-dynamics"
    )

    # -- apply per-cell overrides by key --
    local kv k v
    for kv in "$@"; do
        k="${kv%%=*}"; v="${kv#*=}"
        C["$k"]="$v"
    done

    # -- emit one run per seed --
    local SEED
    for SEED in "${SEEDS[@]}"; do
        local args=()
        for k in "${!C[@]}"; do args+=("${k}=${C[$k]}"); done
        echo "[ablation] launch ${run_name}_s${SEED}  (group=${wandb_group})"
        python3 "${PYTHON_SCRIPT}" "${args[@]}" \
            seed="${SEED}" \
            wandb_group="${wandb_group}" \
            wandb_run_name="${run_name}_s${SEED}" &
    done
}

# `launch_raw` is for cells that must NOT inherit the recursive-anchor overrides
# (e.g. B2 scales model_size and must let the size preset set latent_dim / cycles
# / hidden_size). It bakes only the constant infra keys and passes everything
# else through verbatim, one run per seed.
launch_raw() {
    local run_name="$1"; shift
    local wandb_group="$1"; shift
    local SEED
    for SEED in "${SEEDS[@]}"; do
        echo "[ablation] launch_raw ${run_name}_s${SEED}  (group=${wandb_group})"
        python3 "${PYTHON_SCRIPT}" \
            task="dmcontrol" \
            num_envs=21 \
            obs="state" \
            use_trm_encoder=False \
            use_task_embedding=True \
            enable_wandb=True \
            wandb_project="TRM Dynamics" \
            wandb_entity="trm-dynamics" \
            wandb_group="${wandb_group}" \
            wandb_run_name="${run_name}_s${SEED}" \
            seed="${SEED}" \
            "$@" &
    done
}
