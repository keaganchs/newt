#!/bin/bash
# F6: use_film_dynamics=False @8h4l (SimpleTRM). ISOLATED deep run.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_8h4l_nofilm" "abl_film" use_film_dynamics=False H_cycles=8 L_cycles=4

wait
