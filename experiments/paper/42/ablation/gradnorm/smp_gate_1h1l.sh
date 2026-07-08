#!/bin/bash
# D1: gradnorm logging, gate {off,additive,mlp,swiglu} @1h1l (compile ON; log_trm_gradnorms graph-breaks only the recursion).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_1h1l_gn_off" "abl_gradnorm" use_simple_trm_skip_connections=False H_cycles=1 L_cycles=1 log_trm_gradnorms=True
launch "smp_384ld_1h1l_gn_additive" "abl_gradnorm" use_simple_trm_skip_connections=True simple_trm_skip_type=additive H_cycles=1 L_cycles=1 log_trm_gradnorms=True
launch "smp_384ld_1h1l_gn_mlp" "abl_gradnorm" use_simple_trm_skip_connections=True simple_trm_skip_type=mlp H_cycles=1 L_cycles=1 log_trm_gradnorms=True
launch "smp_384ld_1h1l_gn_swiglu" "abl_gradnorm" use_simple_trm_skip_connections=True simple_trm_skip_type=swiglu H_cycles=1 L_cycles=1 log_trm_gradnorms=True

wait
