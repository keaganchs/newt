#!/bin/bash
# ===========================================================================
# P1 / F3  --  gate x cycles on SimpleTRM, SLOW bucket (4h3l, 8h4l)
# ---------------------------------------------------------------------------
# gate in {off, additive, mlp, swiglu} x cycles in {4h3l, 8h4l}. compile=True.
# Gate "off" x 4h3l == anchor. Same group as the fast half so the full
# gate x cycles surface can be pulled together.
#   group: abl_f3_gate_x_cycles
# 4 gates x 2 cycles = 8 cells x 3 seeds = 24 runs. SLOW (deep recursion).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

CYCLES=("4h3l 4 3" "8h4l 8 4")

for spec in "${CYCLES[@]}"; do
    read -r CYC H L <<< "$spec"
    launch "f3_smp_384ld_${CYC}_gateoff" "abl_f3_gate_x_cycles" \
        H_cycles="$H" L_cycles="$L" use_simple_trm_skip_connections=False
    for SKIP in additive mlp swiglu; do
        launch "f3_smp_384ld_${CYC}_${SKIP}" "abl_f3_gate_x_cycles" \
            H_cycles="$H" L_cycles="$L" \
            use_simple_trm_skip_connections=True simple_trm_skip_type="$SKIP"
    done
done

wait
