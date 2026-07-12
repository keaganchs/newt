#!/bin/bash
# FiLM architecture sweep @16ld: Newt (FiLMDynamics, XL core), SimpleTRM and SRM,
# each with FiLM task conditioning, crossed with film_action_conditioning:
#   taskact  = FiLM conditions on [task_emb, action]
#   taskonly = FiLM conditions on task_emb; the action joins the trunk input
# Recursive cells run the paper depth 4h3l with an L_layers=4 core.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

# Newt baseline dynamics (plain FiLMDynamics, [512, 512] XL core)
launch "newt_16ld_xl_film_taskact"  "abl_film_arch" use_trm_dynamics=None latent_dim=16 hidden_size=16 use_film_dynamics=True xl_dynamics_mlp=True film_action_conditioning=True
launch "newt_16ld_xl_film_taskonly" "abl_film_arch" use_trm_dynamics=None latent_dim=16 hidden_size=16 use_film_dynamics=True xl_dynamics_mlp=True film_action_conditioning=False

# SimpleTRM recursive dynamics
launch "smp_16ld_4h3l_4ll_film_taskact"  "abl_film_arch" use_trm_dynamics=simple latent_dim=16 hidden_size=16 use_film_dynamics=True L_layers=4 H_cycles=4 L_cycles=3 film_action_conditioning=True
launch "smp_16ld_4h3l_4ll_film_taskonly" "abl_film_arch" use_trm_dynamics=simple latent_dim=16 hidden_size=16 use_film_dynamics=True L_layers=4 H_cycles=4 L_cycles=3 film_action_conditioning=False

# SRM recursive dynamics
launch "srm_16ld_4h3l_4ll_film_taskact"  "abl_film_arch" use_trm_dynamics=srm latent_dim=16 hidden_size=16 use_film_dynamics=True L_layers=4 H_cycles=4 L_cycles=3 film_action_conditioning=True
launch "srm_16ld_4h3l_4ll_film_taskonly" "abl_film_arch" use_trm_dynamics=srm latent_dim=16 hidden_size=16 use_film_dynamics=True L_layers=4 H_cycles=4 L_cycles=3 film_action_conditioning=False

wait
