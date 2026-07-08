#!/bin/bash
# F2 compute-matched baseline: Newt MLP + xl_dynamics_mlp x latent_dim.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "newt_xl_16ld" "abl_latent_dim_xl" use_trm_dynamics=None xl_dynamics_mlp=True latent_dim=16 
launch "newt_xl_128ld" "abl_latent_dim_xl" use_trm_dynamics=None xl_dynamics_mlp=True latent_dim=128 
launch "newt_xl_384ld" "abl_latent_dim_xl" use_trm_dynamics=None xl_dynamics_mlp=True latent_dim=384 
launch "newt_xl_512ld" "abl_latent_dim_xl" use_trm_dynamics=None xl_dynamics_mlp=True latent_dim=512 

wait
