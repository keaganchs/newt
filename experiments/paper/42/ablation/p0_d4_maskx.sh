#!/bin/bash
# ===========================================================================
# P0 / D4  --  rrm_mask_x_for_y_update ablation
# ---------------------------------------------------------------------------
# Zero-masking the WM latent x during the y carry update, on/off. True == anchor;
# both cells are kept in one group so the pair is self-contained for plotting.
#   group: abl_d4_maskx
# 2 cells x 3 seeds = 6 runs. MEDIUM (4h3l).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

launch "d4_smp_384ld_4h3l_maskx_on"  "abl_d4_maskx" rrm_mask_x_for_y_update=True
launch "d4_smp_384ld_4h3l_maskx_off" "abl_d4_maskx" rrm_mask_x_for_y_update=False

wait
