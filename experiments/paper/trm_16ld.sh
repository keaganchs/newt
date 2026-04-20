#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running TRM 16ld experiments from ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

sbatch "experiments/paper/trm_16ld/trm_8hs.sh"
sbatch "experiments/paper/trm_16ld/trm_16hs.sh"
sbatch "experiments/paper/trm_16ld/trm_32hs.sh"
