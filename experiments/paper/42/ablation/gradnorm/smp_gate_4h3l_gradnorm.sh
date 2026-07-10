#!/bin/bash
# F3: gate {off,additive,mlp,swiglu} @4h3l (SimpleTRM).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "smp_384ld_4h3l_off" "abl_gradnorm_gate" use_simple_trm_skip_connections=False log_trm_gradnorms=True
launch "smp_384ld_4h3l_additive" "abl_gradnorm_gate" use_simple_trm_skip_connections=True simple_trm_skip_type=additive log_trm_gradnorms=True
launch "smp_384ld_4h3l_mlp" "abl_gradnorm_gate" use_simple_trm_skip_connections=True simple_trm_skip_type=mlp log_trm_gradnorms=True
launch "smp_384ld_4h3l_swiglu" "abl_gradnorm_gate" use_simple_trm_skip_connections=True simple_trm_skip_type=swiglu log_trm_gradnorms=True

wait
