#!/bin/bash
# S2: srm_truncation_length {1,2,3,6,12} @4h3l (SRM).
#
# A100 / 42-style: run in its OWN shell with ONE GPU visible. MPS is on by
# default (pass --disable-mps to turn off). Packs all cells below onto the
# one visible GPU via `&` ... `wait`.
source "$(dirname "$(realpath "$0")")/../_ablation_common.sh"

launch "srm_384ld_4h3l_trunc1" "abl_srm_truncation" use_trm_dynamics=srm srm_truncation_length=1
launch "srm_384ld_4h3l_trunc2" "abl_srm_truncation" use_trm_dynamics=srm srm_truncation_length=2
launch "srm_384ld_4h3l_trunc3" "abl_srm_truncation" use_trm_dynamics=srm srm_truncation_length=3
launch "srm_384ld_4h3l_trunc6" "abl_srm_truncation" use_trm_dynamics=srm srm_truncation_length=6
launch "srm_384ld_4h3l_trunc12" "abl_srm_truncation" use_trm_dynamics=srm srm_truncation_length=12

wait
