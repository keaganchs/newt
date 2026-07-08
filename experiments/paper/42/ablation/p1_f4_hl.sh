#!/bin/bash
# ===========================================================================
# P1 / F4  --  H-vs-L disentanglement on SimpleTRM
# ---------------------------------------------------------------------------
# (H_cycles, L_cycles) in {(4,1), (2,2), (1,4)} -- same total-ish budget, moved
# between the outer (H) and inner (L) loop, to see which axis matters. Anchor
# otherwise (384ld, simple, simnorm, no skip, no film).
#   group: abl_f4_h_vs_l
# 3 cells x 3 seeds = 9 runs. MEDIUM.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

HL=("4h1l 4 1" "2h2l 2 2" "1h4l 1 4")

for spec in "${HL[@]}"; do
    read -r CYC H L <<< "$spec"
    launch "f4_smp_384ld_${CYC}" "abl_f4_h_vs_l" H_cycles="$H" L_cycles="$L"
done

wait
