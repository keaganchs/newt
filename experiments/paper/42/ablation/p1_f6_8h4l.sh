#!/bin/bash
# ===========================================================================
# P1 / F6  --  FiLM conditioning, 8h4l  (SLOW bucket)
# ===========================================================================
# use_film_dynamics in {False, True} at 8h4l. Same group as the 1h1l half.
#   group: abl_f6_film
# 2 cells x 3 seeds = 6 runs. SLOW (deep recursion).
# ===========================================================================
source "$(dirname "$(realpath "$0")")/_ablation_common.sh"

launch "f6_smp_384ld_8h4l_nofilm" "abl_f6_film" H_cycles=8 L_cycles=4 use_film_dynamics=False
launch "f6_smp_384ld_8h4l_film"   "abl_f6_film" H_cycles=8 L_cycles=4 use_film_dynamics=True

wait
