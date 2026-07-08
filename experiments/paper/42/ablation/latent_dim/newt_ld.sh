#!/bin/bash
# F1 headline: Newt (MLP dynamics) x latent_dim {16,128,384,512}.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "newt_16ld" "abl_latent_dim" use_trm_dynamics=None latent_dim=16 
launch "newt_128ld" "abl_latent_dim" use_trm_dynamics=None latent_dim=128 
launch "newt_384ld" "abl_latent_dim" use_trm_dynamics=None latent_dim=384 
launch "newt_512ld" "abl_latent_dim" use_trm_dynamics=None latent_dim=512 

wait
