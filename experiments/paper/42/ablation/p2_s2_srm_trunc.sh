#!/bin/bash
# ===========================================================================
# P2 / S2  --  SRM truncated-BPTT length sweep
# ---------------------------------------------------------------------------
# srm_truncation_length in {1,2,3,6,12} (number of recursion steps carrying
# gradients) at a fixed, reasonably deep recursion (4h3l = 12 inner steps, so 12
# == full-gradient BPTT and 1 == maximally truncated). Truncation is meaningless
# at 1h1l, hence the deep fixed budget. SRM anchor (use_trm_dynamics=srm), 384ld.
# Skip-type comparison is folded into F3 and NOT repeated here.
#   group: abl_s2_srm_truncation
# 5 cells x 3 seeds = 15 runs. SLOW-ish (deep recursion).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for T in 1 2 3 6 12; do
    launch "s2_srm_384ld_4h3l_trunc${T}" "abl_s2_srm_truncation" \
        use_trm_dynamics=srm H_cycles=4 L_cycles=3 srm_truncation_length="$T"
done

wait
