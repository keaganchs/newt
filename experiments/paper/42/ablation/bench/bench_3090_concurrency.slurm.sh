#!/bin/bash
#SBATCH --job-name=bench3090
#SBATCH --output=log/out_and_err_%j.txt
#SBATCH --error=log/out_and_err_%j.txt
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=9
#SBATCH --mem-per-cpu=5500
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1

# Concurrency benchmark for the 3090 cluster: how many low-recursion Newt runs
# actually fit on one RTX 3090? You found 3 (1 cell x 3 seeds) fit and more
# "consistently crashed" -- this measures aggregate throughput vs. concurrency so
# we can see whether >3 is worthwhile (and whether the crashes are OOM or the
# CPU/env bottleneck) before committing the sweep.
#
# Submit from the repo root:   sbatch experiments/paper/42/ablation/bench/bench_3090_concurrency.slurm.sh
# Reads the machine-readable summary + scheduler hint printed by Script A.

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate newt

cd "$SLURM_SUBMIT_DIR"

# small = latent_dim<=384 SimpleTRM at 1h1l (the low-recursion 3090 workload).
# --compile matches production kernels. Sweep 1..6 concurrent runs on the single
# visible 3090; 21 env workers/run means level 6 = 126 env procs (< default cap).
python3 benchmarks/bench_concurrency.py \
    --config small \
    --h 1 --l 1 \
    --num-envs 21 \
    --levels "1,2,3,4,5,6" \
    --max-concurrency 6 \
    --duration 12 \
    --compile

echo "[bench] done. If throughput keeps rising past 3 runs, raise the per-job cell"
echo "        count on the 3090 SBATCH scripts; if it plateaus/OOMs at 3, keep 3."
