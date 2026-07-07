#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# NVIDIA CUDA MPS control helper for packing many small Newt runs on one GPU.
#
# Why: the concurrency benchmark shows the runs are latency/overhead-bound -- the
# GPU sits at ~20-30% util even with 8 concurrent runs because each process gets a
# time-sliced CUDA context and its many tiny kernels serialise. MPS (Multi-Process
# Service) lets kernels from DIFFERENT processes execute concurrently on the GPU's
# SMs, which is exactly what a fleet of tiny-model RL runs needs -> higher aggregate
# throughput and more runs per GPU.
#
# This starts a *per-user* MPS daemon (no root needed) scoped to this shell via
# private pipe/log dirs. Every CUDA process launched from a shell that has sourced
# these env vars will route through MPS. Prefer MPS over MIG here: MIG caps at 7
# isolated slices and can't oversubscribe; you want spatial sharing.
#
# Usage:
#   source ./mps_control.sh start     # start daemon + export env into THIS shell
#   ./mps_control.sh status
#   source ./mps_control.sh stop      # stop daemon + unset env
#
# NOTE: use `source` for start/stop so the CUDA_MPS_* env vars land in your shell
# (and thus in the benchmark / train.py processes you launch afterwards). On a SLURM
# cluster you may instead need MPS enabled at the job level (e.g. `srun --gpu-mps`);
# check your site docs if the daemon won't start.
# ---------------------------------------------------------------------------

_MPS_ROOT="${CUDA_MPS_ROOT:-/tmp/newt-mps-$USER}"
export CUDA_MPS_PIPE_DIRECTORY="${_MPS_ROOT}/pipe"
export CUDA_MPS_LOG_DIRECTORY="${_MPS_ROOT}/log"

_cmd="${1:-status}"

case "$_cmd" in
  start)
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if pgrep -u "$USER" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "[mps] daemon already running for $USER"
    else
      # Optionally cap per-client SM usage so one run can't hog all SMs, e.g.:
      #   export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=25
      nvidia-cuda-mps-control -d && echo "[mps] daemon started"
    fi
    echo "[mps] CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "[mps] Launch your runs from THIS shell so they route through MPS."
    echo "[mps] (If you ran this without 'source', the env vars won't persist -- re-run: source $0 start)"
    ;;
  stop)
    if pgrep -u "$USER" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo quit | nvidia-cuda-mps-control && echo "[mps] daemon stopped"
    else
      echo "[mps] no daemon running for $USER"
    fi
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    ;;
  status)
    if pgrep -u "$USER" -f nvidia-cuda-mps-control >/dev/null 2>&1; then
      echo "[mps] daemon RUNNING for $USER (pipe: $CUDA_MPS_PIPE_DIRECTORY)"
      echo get_server_list | nvidia-cuda-mps-control 2>/dev/null || true
    else
      echo "[mps] daemon NOT running for $USER"
    fi
    ;;
  *)
    echo "usage: source $0 {start|stop|status}"
    ;;
esac
