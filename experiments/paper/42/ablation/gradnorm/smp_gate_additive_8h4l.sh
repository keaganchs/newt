#!/bin/bash
# D1: gradnorm logging, gate additive @8h4l (log_trm_gradnorms). ISOLATED deep run.
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_8h4l_gn_additive" "abl_gradnorm" use_simple_trm_skip_connections=True simple_trm_skip_type=additive H_cycles=8 L_cycles=4 log_trm_gradnorms=True

wait
