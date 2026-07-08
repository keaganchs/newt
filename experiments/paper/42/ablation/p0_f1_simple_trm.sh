#!/bin/bash
# ===========================================================================
# P0 / F1 (headline)  --  SimpleTRM + TRM recursive columns, MEDIUM bucket
# ---------------------------------------------------------------------------
# F1 axis: use_trm_dynamics in {simple, trm} x latent_dim in {16,128,384,512},
# at the anchor recursion depth 4h3l. hidden_size is tied to latent_dim so the
# recursion width scales with the latent under one knob (num_heads stays at the
# S preset =2, which divides all four latent_dims).
#   group: abl_f1_ld_x_dyn
# 8 cells x 3 seeds = 24 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for LD in 16 128 384 512; do
    launch "f1_smp_${LD}ld_4h3l" "abl_f1_ld_x_dyn" \
        use_trm_dynamics=simple latent_dim="$LD" hidden_size="$LD"
done

for LD in 16 128 384 512; do
    launch "f1_trm_${LD}ld_4h3l" "abl_f1_ld_x_dyn" \
        use_trm_dynamics=trm latent_dim="$LD" hidden_size="$LD"
done

wait
