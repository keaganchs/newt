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

# ---- MPS (on by default) --------------------------------------------------
# Each A100 script is launched in its OWN shell with a single GPU visible, then
# packs several runs onto that GPU; MPS lets their kernels share the SMs. It is
# enabled by default here. Disable per invocation with either:
#     DISABLE_MPS=1 bash <script.sh>
#     bash <script.sh> --disable-mps
for _arg in "$@"; do [ "${_arg}" = "--disable-mps" ] && DISABLE_MPS=1; done
if [ "${DISABLE_MPS:-0}" != "1" ]; then
    _MPS_HELPER="${PROJECT_ROOT}/benchmarks/mps_control.sh"
    if [ -f "${_MPS_HELPER}" ]; then
        # shellcheck disable=SC1090
        source "${_MPS_HELPER}" start || echo "[ablation] MPS start failed; continuing without it."
    else
        echo "[ablation] MPS helper not found at ${_MPS_HELPER}; continuing without MPS."
    fi
else
    echo "[ablation] MPS disabled (--disable-mps / DISABLE_MPS=1)."
fi

# ---- seeds ----------------------------------------------------------------
SEEDS=(0 1 2)

# ---- launcher -------------------------------------------------------------
# usage: launch <run_name> <plan_tag> [key=val ...]
#   <run_name>    encodes axis + key hyperparameters (the CELL). "_s<SEED>" is
#                 appended for the wandb run name.
#   <plan_tag>    the experiment/plan this cell belongs to (e.g. abl_f5_regularization).
#   key=val ...   hydra overrides that move this cell off the anchor.
#
# WANDB GROUPING: the wandb group is set PER CELL to "<plan_tag>/<run_name>", so
# the 3 seeds of a cell share a group (wandb's "group by group" then aggregates
# them into one mean+/-band line) while every cell is a DISTINCT group. To pull a
# whole plan for a plot, filter groups by the "<plan_tag>/" prefix. (Setting the
# group to <plan_tag> alone -- as before -- collapsed every cell into one line.)
#
# Overrides are merged into the anchor BY KEY (so we never emit a duplicate
# hydra key, which hydra rejects). One backgrounded run is launched per seed.
launch() {
    local run_name="$1"; shift
    local plan_tag="$1"; shift
    local cell_group="${plan_tag}/${run_name}"

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

    # -- bf16 for the recursive dynamics (SimpleTRM/TRM/SRM): ~2x on the memory-bound
    #    planner + update (bf16 tensor cores; master weights & env I/O stay fp32). Skipped
    #    for the Newt MLP baseline (use_trm_dynamics=None), which is benchmarked separately.
    #    A cell may still override amp_dtype explicitly (the merge above wins). --
    case "${C[use_trm_dynamics]}" in
        trm) [ -z "${C[amp_dtype]:-}" ] && C[amp_dtype]="bfloat16" ;;
    esac
    
    # case "${C[use_trm_dynamics]}" in
    #     simple|trm|srm) [ -z "${C[amp_dtype]:-}" ] && C[amp_dtype]="bfloat16" ;;
    # esac


    # -- emit one run per seed --
    local SEED
    for SEED in "${SEEDS[@]}"; do
        local args=()
        for k in "${!C[@]}"; do args+=("${k}=${C[$k]}"); done
        echo "[ablation] launch ${run_name}_s${SEED}  (group=${cell_group})"
        python3 "${PYTHON_SCRIPT}" "${args[@]}" \
            seed="${SEED}" \
            wandb_group="${cell_group}" \
            wandb_run_name="${run_name}_s${SEED}" &
    done
}

# `launch_raw` is for cells that must NOT inherit the recursive-anchor overrides
# (e.g. B2 scales model_size and must let the size preset set latent_dim / cycles
# / hidden_size). It bakes only the constant infra keys and passes everything
# else through verbatim, one run per seed.
launch_raw() {
    local run_name="$1"; shift
    local plan_tag="$1"; shift
    local cell_group="${plan_tag}/${run_name}"    # per-cell group; see launch() note
    local SEED
    for SEED in "${SEEDS[@]}"; do
        echo "[ablation] launch_raw ${run_name}_s${SEED}  (group=${cell_group})"
        python3 "${PYTHON_SCRIPT}" \
            task="dmcontrol" \
            num_envs=21 \
            obs="state" \
            use_trm_encoder=False \
            use_task_embedding=True \
            enable_wandb=True \
            wandb_project="TRM Dynamics" \
            wandb_entity="trm-dynamics" \
            wandb_group="${cell_group}" \
            wandb_run_name="${run_name}_s${SEED}" \
            seed="${SEED}" \
            "$@" &
    done
}
