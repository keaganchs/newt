#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running TRM 128ld experiments from ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

sbatch "experiments/paper/trm_128ld/trm_64hs.sh"
sbatch "experiments/paper/trm_128ld/trm_128hs.sh"
sbatch "experiments/paper/trm_128ld/trm_256hs.sh"
