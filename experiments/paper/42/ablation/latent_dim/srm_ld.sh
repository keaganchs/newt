#!/bin/bash
# F1 headline: SRM dynamics x latent_dim @4h3l (truncation=3).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "srm_16ld_4h3l" "abl_latent_dim" use_trm_dynamics=srm srm_truncation_length=3 latent_dim=16 hidden_size=16
launch "srm_128ld_4h3l" "abl_latent_dim" use_trm_dynamics=srm srm_truncation_length=3 latent_dim=128 hidden_size=128
launch "srm_384ld_4h3l" "abl_latent_dim" use_trm_dynamics=srm srm_truncation_length=3 latent_dim=384 hidden_size=384
launch "srm_512ld_4h3l" "abl_latent_dim" use_trm_dynamics=srm srm_truncation_length=3 latent_dim=512 hidden_size=512

wait
