#!/bin/bash
# ===========================================================================
# P1 / F3  --  gate x cycles on SimpleTRM, FAST bucket (1h1l, 2h2l)
# ---------------------------------------------------------------------------
# gate in {off, additive, mlp, swiglu} x cycles in {1h1l, 2h2l}. compile=True
# (the instrumented / gradnorm slice is D1, not repeated here). Gate "off" ==
# anchor (skip connections disabled; skip_type is then irrelevant).
#   group: abl_f3_gate_x_cycles
# 4 gates x 2 cycles = 8 cells x 3 seeds = 24 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

# cycles as "name H L"
CYCLES=("1h1l 1 1" "2h2l 2 2")

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
