#!/bin/bash
# F1 headline: SimpleTRM dynamics x latent_dim @4h3l.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_16ld_4h3l" "abl_latent_dim" latent_dim=16 
launch "smp_128ld_4h3l" "abl_latent_dim" latent_dim=128 
launch "smp_384ld_4h3l" "abl_latent_dim" latent_dim=384 
launch "smp_512ld_4h3l" "abl_latent_dim" latent_dim=512 

wait
