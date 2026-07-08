#!/bin/bash
# ===========================================================================
# P1 / F6  --  FiLM conditioning, 1h1l  (FAST bucket)
# ===========================================================================
# use_film_dynamics in {False, True} at 1h1l. False == anchor. Split from the
# 8h4l half so each batch finishes coherently.
#   group: abl_f6_film
# 2 cells x 3 seeds = 6 runs.
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

launch "f6_smp_384ld_1h1l_nofilm" "abl_f6_film" H_cycles=1 L_cycles=1 use_film_dynamics=False
launch "f6_smp_384ld_1h1l_film"   "abl_f6_film" H_cycles=1 L_cycles=1 use_film_dynamics=True

wait
