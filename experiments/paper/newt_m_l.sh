#!/bin/bash

echo "Running Newt M experiments"

sbatch "$SLURM_SUBMIT_DIR/experiments/paper/newt_m_l/newt_m.sh"
# sbatch "$SLURM_SUBMIT_DIR/experiments/paper/newt_m_l/newt_l.sh"