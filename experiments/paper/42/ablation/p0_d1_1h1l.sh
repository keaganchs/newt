#!/bin/bash
# ===========================================================================
# P0 / D1  --  gradnorm instrumentation, 1h1l  (log_trm_gradnorms=True)
# ---------------------------------------------------------------------------
# Captures per-cycle gradnorms, ||dz|| and inter-cycle cosine similarity on the
# SimpleTRM dynamics across the gate variants, at 1h1l. THESE ARE THE ONLY RUNS
# WITH log_trm_gradnorms=True -> torch.compile is disabled on the inner apply,
# so they are much slower and MUST live in their own batches (not mixed with the
# compiled runs). Gate "off" == anchor (skip connections disabled).
#   group: abl_d1_gradnorm
# 4 cells x 3 seeds = 12 runs. FAST-of-the-slow (1h1l).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

# gate off (anchor)
launch "d1_smp_384ld_1h1l_gateoff_gradnorm" "abl_d1_gradnorm" \
    H_cycles=1 L_cycles=1 log_trm_gradnorms=True \
    use_simple_trm_skip_connections=False

# gate on x {additive, mlp, swiglu}
for SKIP in additive mlp swiglu; do
    launch "d1_smp_384ld_1h1l_${SKIP}_gradnorm" "abl_d1_gradnorm" \
        H_cycles=1 L_cycles=1 log_trm_gradnorms=True \
        use_simple_trm_skip_connections=True simple_trm_skip_type="$SKIP"
done

wait
