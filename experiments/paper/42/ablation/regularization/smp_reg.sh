#!/bin/bash
# F5: wm_regularization {simnorm,sigreg,none} x latent_dim {16,384} (SimpleTRM @4h3l).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_16ld_4h3l_simnorm" "abl_regularization" latent_dim=16 hidden_size=16 wm_regularization_type=simnorm
launch "smp_16ld_4h3l_sigreg" "abl_regularization" latent_dim=16 hidden_size=16 wm_regularization_type=sigreg
launch "smp_16ld_4h3l_none" "abl_regularization" latent_dim=16 hidden_size=16 wm_regularization_type=none
launch "smp_384ld_4h3l_simnorm" "abl_regularization" latent_dim=384 hidden_size=384 wm_regularization_type=simnorm
launch "smp_384ld_4h3l_sigreg" "abl_regularization" latent_dim=384 hidden_size=384 wm_regularization_type=sigreg
launch "smp_384ld_4h3l_none" "abl_regularization" latent_dim=384 hidden_size=384 wm_regularization_type=none

wait
