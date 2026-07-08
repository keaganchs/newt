#!/bin/bash
# ===========================================================================
# P0 / D1  --  gradnorm instrumentation, 8h4l  (log_trm_gradnorms=True)
# ---------------------------------------------------------------------------
# Same as p0_d1_1h1l.sh but at the deep 8h4l depth, where the per-cycle gradnorm
# / ||dz|| / cosine diagnostics matter most (this is where the SwiGLU skip was
# found to blow up at init). log_trm_gradnorms=True disables compile -> SLOWEST
# batch in the whole plan; keep it isolated.  Gate "off" == anchor.
#   group: abl_d1_gradnorm
# 4 cells x 3 seeds = 12 runs. SLOWEST bucket.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

# gate off (anchor)
launch "d1_smp_384ld_8h4l_gateoff_gradnorm" "abl_d1_gradnorm" \
    H_cycles=8 L_cycles=4 log_trm_gradnorms=True \
    use_simple_trm_skip_connections=False

# gate on x {additive, mlp, swiglu}
for SKIP in additive mlp swiglu; do
    launch "d1_smp_384ld_8h4l_${SKIP}_gradnorm" "abl_d1_gradnorm" \
        H_cycles=8 L_cycles=4 log_trm_gradnorms=True \
        use_simple_trm_skip_connections=True simple_trm_skip_type="$SKIP"
done

wait
