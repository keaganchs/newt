#!/bin/bash
# F1 headline: TRM dynamics x latent_dim @4h3l.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "trm_16ld_4h3l" "abl_latent_dim" use_trm_dynamics=trm latent_dim=16 hidden_size=128 num_state_obs_per_token=128 amp_dtype="bfloat16"
launch "trm_128ld_4h3l" "abl_latent_dim" use_trm_dynamics=trm latent_dim=128 hidden_size=128 num_state_obs_per_token=128 amp_dtype="bfloat16"
launch "trm_384ld_4h3l" "abl_latent_dim" use_trm_dynamics=trm latent_dim=384 hidden_size=128 num_state_obs_per_token=128 amp_dtype="bfloat16"
launch "trm_512ld_4h3l" "abl_latent_dim" use_trm_dynamics=trm latent_dim=512 hidden_size=128 num_state_obs_per_token=128 amp_dtype="bfloat16"

wait
