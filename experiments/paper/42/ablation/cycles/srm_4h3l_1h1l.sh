#!/bin/bash
# Cycles sweep: SRM @1h1l and @4h3l, latent_dim=384. Completes the
# abl_cycles_srm axis (2h2l and 8h4l run on the 3090 cluster via the
# srm_*.slurm.sh scripts in this dir, same config).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "srm_384ld_1h1l" "abl_cycles_srm" use_trm_dynamics=srm H_cycles=1 L_cycles=1 srm_truncation_length=1
launch "srm_384ld_4h3l" "abl_cycles_srm" use_trm_dynamics=srm H_cycles=4 L_cycles=3 srm_truncation_length=3

wait
