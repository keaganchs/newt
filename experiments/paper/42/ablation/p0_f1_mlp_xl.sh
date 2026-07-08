#!/bin/bash
# ===========================================================================
# P0 / F1 (headline) + F2 xl baseline  --  MLP-dynamics column, FAST bucket
# ---------------------------------------------------------------------------
# F1 axis: use_trm_dynamics=None (plain MLP dynamics) x latent_dim in {16,128,384,512}.
# Plus the compute-matched MLP baseline (xl_dynamics_mlp=True) at the same four
# latent_dims -- this IS the F2 sweep (F2's requested {16,384} is a subset; we run
# all four so the headline has a full MLP-XL column). xl_dynamics_mlp ONLY affects
# the default MLP dynamics, so it is combined with use_trm_dynamics=None only.
#
# No recursion here (H/L unused by the MLP dynamics) -> fastest bucket.
#   groups: abl_f1_ld_x_dyn   (MLP column of the headline)
#           abl_f2_mlp_xl      (compute-matched / capacity-control MLP baseline)
# 8 cells x 3 seeds = 24 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for LD in 16 128 384 512; do
    launch "f1_mlp_${LD}ld"   "abl_f1_ld_x_dyn"  use_trm_dynamics=None latent_dim="$LD" hidden_size="$LD"
done

for LD in 16 128 384 512; do
    launch "f2_mlpxl_${LD}ld" "abl_f2_mlp_xl"    use_trm_dynamics=None latent_dim="$LD" hidden_size="$LD" xl_dynamics_mlp=True
done

wait
