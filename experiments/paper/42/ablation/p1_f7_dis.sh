#!/bin/bash
# ===========================================================================
# P1 / F7  --  Deep Improvement Supervision (DIS) x latent_dim
# ---------------------------------------------------------------------------
# use_dis_loss in {False, True} x latent_dim in {16, 384}; when ON, the secondary
# dis_schedule in {linear, cosine} (dis_loss_coef stays at the 1.0 default). OFF
# == anchor exactly (dis_schedule is then irrelevant, so it is not varied).
# OOD-cycle behaviour of DIS-trained models is measured by D3 (eval-only), not here.
#   group: abl_f7_dis
# off: 2 cells + on: 2 ld x 2 sched = 4 cells  ->  6 cells x 3 seeds = 18 runs. MEDIUM.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

# DIS off (anchor)
for LD in 16 384; do
    launch "f7_smp_${LD}ld_4h3l_dis_off" "abl_f7_dis" \
        latent_dim="$LD" hidden_size="$LD" use_dis_loss=False
done

# DIS on x schedule
for LD in 16 384; do
    for SCHED in linear cosine; do
        launch "f7_smp_${LD}ld_4h3l_dis_${SCHED}" "abl_f7_dis" \
            latent_dim="$LD" hidden_size="$LD" \
            use_dis_loss=True dis_schedule="$SCHED"
    done
done

wait
