#!/bin/bash
# ===========================================================================
# P3 / U1  --  qualitative rollout videos  (EVAL-ONLY, no retraining)
# ===========================================================================
# save_video=True requires env_mode=sync. Loads a few trained F1 checkpoints and
# records evaluation rollouts. Same eval-only mechanism as D3 (load checkpoint,
# run the step-0 eval, exit). Kept single-process-ish and small; videos are heavy,
# so DO NOT pack this batch as densely as the training batches.
#
# >>> DEPENDENCY: needs trained F1 checkpoints. Fill CHECKPOINTS (see D3 header
#     for the "label | path | arch overrides" format).
#
#   group: abl_u1_video
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

GROUP="abl_u1_video"
EVAL_EPISODES=3

# --- FILL ME IN --------------------------------------------------------------
CHECKPOINTS=(
    # "f1_smp_384ld_4h3l_s0 | /pfss/.../f1_smp_384ld_4h3l_s0/0/models/<step>.pt | use_trm_dynamics=simple latent_dim=384 hidden_size=384 H_cycles=4 L_cycles=3"
    # "f1_mlp_384ld_s0      | /pfss/.../f1_mlp_384ld_s0/0/models/<step>.pt      | use_trm_dynamics=None   latent_dim=384 hidden_size=384"
)
# -----------------------------------------------------------------------------

if [ "${#CHECKPOINTS[@]}" -eq 0 ]; then
    echo "[U1] CHECKPOINTS is empty -- fill it with trained F1 checkpoints first (see header). Aborting."
    exit 1
fi

for entry in "${CHECKPOINTS[@]}"; do
    IFS='|' read -r LABEL CKPT ARCH <<< "$entry"
    LABEL="$(echo "$LABEL" | xargs)"; CKPT="$(echo "$CKPT" | xargs)"; ARCH="$(echo "$ARCH" | xargs)"
    echo "[U1] video ${LABEL}"
    python3 "${PYTHON_SCRIPT}" \
        task="dmcontrol" num_envs=21 obs="state" \
        use_trm_encoder=False use_task_embedding=True model_size="S" \
        enable_wandb=True wandb_project="TRM Dynamics" wandb_entity="trm-dynamics" \
        wandb_group="${GROUP}" \
        wandb_run_name="u1_${LABEL}_video" \
        checkpoint="${CKPT}" \
        mpc=True steps=1 save_agent=False eval_episodes="${EVAL_EPISODES}" \
        save_video=True env_mode=sync \
        ${ARCH} &
done

wait
