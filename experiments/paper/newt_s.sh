#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running Newt S experiments from ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

sbatch "experiments/paper/newt_s/newt_s_128ld.sh"
sbatch "experiments/paper/newt_s/newt_s.sh"
sbatch "experiments/paper/newt_s/newt_s_512ld.sh"
