#!/bin/bash
# ===========================================================================
# P2 / B2  --  Newt model-size baselines (M, L)
# ---------------------------------------------------------------------------
# The standard Newt baseline (plain MLP dynamics, no TRM/SimpleTRM/SRM) at
# model_size M and L. This is a size baseline, NOT a slice at the recursive
# anchor, so it uses launch_raw: model_size drives enc_dim / mlp_dim / num_q /
# num_enc_layers via the preset, and latent_dim is pinned to 128 (matching the
# existing newt_m_l scripts) so only the surrounding MLP capacity scales.
#   group: abl_b2_model_size
# 2 cells x 3 seeds = 6 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for MS in M L; do
    launch_raw "b2_newt_${MS,,}_128ld" "abl_b2_model_size" \
        use_trm_dynamics=False \
        model_size="$MS" \
        latent_dim=128
done

wait
