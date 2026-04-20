#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running Newt M experiments from ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

sbatch "experiments/paper/newt_m_l/newt_m.sh"
sbatch "experiments/paper/newt_m_l/newt_l.sh"
