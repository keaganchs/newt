#!/bin/bash
# F4: (H,L) in {(4,1),(2,2),(1,4)} at fixed budget (SimpleTRM).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_4h1l" "abl_h_vs_l" H_cycles=4 L_cycles=1
launch "smp_384ld_2h2l" "abl_h_vs_l" H_cycles=2 L_cycles=2
launch "smp_384ld_1h4l" "abl_h_vs_l" H_cycles=1 L_cycles=4

wait
