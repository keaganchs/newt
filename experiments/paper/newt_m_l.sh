#!/bin/bash

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"

echo "Running Newt M experiments from ${PROJECT_ROOT}"

sbatch "${PROJECT_ROOT}/experiments/paper/newt_m_l/newt_m.sh"
sbatch "${PROJECT_ROOT}/experiments/paper/newt_m_l/newt_l.sh"
