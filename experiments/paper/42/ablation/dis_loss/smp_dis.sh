#!/bin/bash
# F7: use_dis_loss off/on x latent_dim {16,384} (+ dis_schedule when on).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_16ld_4h3l_dis_off" "abl_dis_loss" latent_dim=16 hidden_size=16 use_dis_loss=False
launch "smp_384ld_4h3l_dis_off" "abl_dis_loss" latent_dim=384 hidden_size=384 use_dis_loss=False
launch "smp_16ld_4h3l_dis_linear" "abl_dis_loss" latent_dim=16 hidden_size=16 use_dis_loss=True dis_schedule=linear
launch "smp_16ld_4h3l_dis_cosine" "abl_dis_loss" latent_dim=16 hidden_size=16 use_dis_loss=True dis_schedule=cosine
launch "smp_384ld_4h3l_dis_linear" "abl_dis_loss" latent_dim=384 hidden_size=384 use_dis_loss=True dis_schedule=linear
launch "smp_384ld_4h3l_dis_cosine" "abl_dis_loss" latent_dim=384 hidden_size=384 use_dis_loss=True dis_schedule=cosine

wait
