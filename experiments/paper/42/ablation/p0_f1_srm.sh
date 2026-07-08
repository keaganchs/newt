#!/bin/bash
# ===========================================================================
# P0 / F1 (headline)  --  SRM recursive column, MEDIUM bucket
# ---------------------------------------------------------------------------
# F1 axis: use_trm_dynamics=srm x latent_dim in {16,128,384,512}, at the anchor
# depth 4h3l. SRM uses truncated BPTT; srm_truncation_length=3 is the S default
# (the truncation sweep itself is S2). hidden_size tied to latent_dim as for trm.
#   group: abl_f1_ld_x_dyn
# 4 cells x 3 seeds = 12 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for LD in 16 128 384 512; do
    launch "f1_srm_${LD}ld_4h3l" "abl_f1_ld_x_dyn" \
        use_trm_dynamics=srm latent_dim="$LD" hidden_size="$LD" srm_truncation_length=3
done

wait
