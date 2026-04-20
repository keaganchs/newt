#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running TRM 384ld experiments from ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

sbatch "experiments/paper/trm_384ld/trm_64hs.sh"
sbatch "experiments/paper/trm_384ld/trm_64hs_16opt.sh"
sbatch "experiments/paper/trm_384ld/trm_128hs.sh"
sbatch "experiments/paper/trm_384ld/trm_128hs_32opt.sh"
sbatch "experiments/paper/trm_384ld/trm_256hs.sh"