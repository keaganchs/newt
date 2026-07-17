#!/bin/bash
# B2: Newt MLP baseline at model_size {M,L} (use_trm_dynamics=None, latent_dim=128).
# Uses launch_raw so the size preset -- not the recursive anchor -- sets enc/mlp/num_q.
#
# A100 / 42-style; own shell, one GPU, MPS default.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch_raw "newt_m" "abl_model_size" model_size=M use_trm_dynamics=None latent_dim=512
launch_raw "newt_l" "abl_model_size" model_size=L use_trm_dynamics=None latent_dim=512

wait
