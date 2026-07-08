#!/bin/bash
# ===========================================================================
# P3 / D3  --  inference-time cycle scaling  (EVAL-ONLY, no retraining)
# ===========================================================================
# Loads TRAINED F1/F3 checkpoints and re-evaluates them while sweeping the
# planning-time recursion depth planning_H_cycles in {1,2,4,8} (and, where you
# want it, planning_L_cycles). This uses the planning-depth override that decouples
# MPPI-rollout depth from the trained depth -- so it is genuinely eval-only and
# cheap. There is no separate eval entry point in this repo, so we load the
# checkpoint and let the step-0 evaluation run, then exit immediately (steps=1,
# save_agent=False).
#
# >>> DEPENDENCY: this batch cannot run until the referenced F1/F3 runs have
#     produced checkpoints. Fill in CHECKPOINTS below (it is empty by design;
#     the script aborts if you don't). Each entry is:
#         "label | /abs/path/to/model.pt | <arch overrides matching that run>"
#     The arch overrides MUST reproduce the architecture the checkpoint was
#     trained with (dynamics type, latent_dim, hidden_size, trained H/L, gate,
#     film, reg) so the model builds identically before the weights load.
#
#   group: abl_d3_planning_cycles
# ===========================================================================
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

GROUP="abl_d3_planning_cycles"
PLANNING_H=(1 2 4 8)          # inference-time H_cycles to sweep
EVAL_EPISODES=10              # more eval episodes than training default for a clean number

# --- FILL ME IN (see header) -------------------------------------------------
CHECKPOINTS=(
    # "f1_smp_384ld_4h3l_s0 | /pfss/.../logs/dmcontrol/f1_smp_384ld_4h3l_s0/0/models/<step>.pt | use_trm_dynamics=simple latent_dim=384 hidden_size=384 H_cycles=4 L_cycles=3"
    # "f1_smp_16ld_4h3l_s0  | /pfss/.../f1_smp_16ld_4h3l_s0/0/models/<step>.pt          | use_trm_dynamics=simple latent_dim=16  hidden_size=16  H_cycles=4 L_cycles=3"
    # "f3_smp_384ld_8h4l_swiglu_s0 | /pfss/.../f3_smp_384ld_8h4l_swiglu_s0/0/models/<step>.pt | use_trm_dynamics=simple latent_dim=384 hidden_size=384 H_cycles=8 L_cycles=4 use_simple_trm_skip_connections=True simple_trm_skip_type=swiglu"
)
# -----------------------------------------------------------------------------

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "[D3] CHECKPOINTS is empty -- fill it with trained F1/F3 checkpoints first (see header). Aborting."
    exit 1
fi

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r LABEL CKPT ARCH <<< "$entry"
    LABEL="$(echo "$LABEL" | xargs)"; CKPT="$(echo "$CKPT" | xargs)"; ARCH="$(echo "$ARCH" | xargs)"
    for PH in "${PLANNING_H[@]}"; do
        echo "[D3] eval ${LABEL} @ planning_H_cycles=${PH}"
        RUN="d3_${LABEL}_planH${PH}"          # per-cell run + group (see _ablation_common.sh)
        python3 "${PYTHON_SCRIPT}" \
            task="dmcontrol" num_envs=21 obs="state" \
            use_trm_encoder=False use_task_embedding=True model_size="S" \
            enable_wandb=True wandb_project="TRM Dynamics" wandb_entity="trm-dynamics" \
            wandb_group="${GROUP}/${RUN}" \
            wandb_run_name="${RUN}" \
            checkpoint="${CKPT}" \
            mpc=True steps=1 save_agent=False eval_episodes="${EVAL_EPISODES}" \
            planning_H_cycles="${PH}" \
            ${ARCH} &
    done
done

wait
