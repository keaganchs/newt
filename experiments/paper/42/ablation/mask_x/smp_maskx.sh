#!/bin/bash
# D4: rrm_mask_x_for_y_update {True,False} (SimpleTRM @4h3l).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_4h3l_maskx_true" "abl_mask_x" rrm_mask_x_for_y_update=True
launch "smp_384ld_4h3l_maskx_false" "abl_mask_x" rrm_mask_x_for_y_update=False

wait
