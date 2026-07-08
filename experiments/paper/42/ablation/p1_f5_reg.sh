#!/bin/bash
# ===========================================================================
# P1 / F5  --  latent regularization x latent_dim on SimpleTRM
# ---------------------------------------------------------------------------
# wm_regularization_type in {simnorm, sigreg, none} x latent_dim in {16, 384}.
# The three are mutually exclusive: simnorm = simplicial embedding activation;
# sigreg = SIGReg loss (adds sigreg_coef, DISABLES SimNorm); none = latent left
# unconstrained. simnorm x 384 == anchor.
#   group: abl_f5_regularization
# 3 reg x 2 ld = 6 cells x 3 seeds = 18 runs. MEDIUM (4h3l).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

for LD in 16 384; do
    for REG in simnorm sigreg none; do
        launch "f5_smp_${LD}ld_4h3l_${REG}" "abl_f5_regularization" \
            latent_dim="$LD" hidden_size="$LD" wm_regularization_type="$REG"
    done
done

wait
