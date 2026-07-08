#!/bin/bash
# ===========================================================================
# Ablation runner (42 style) -- schedules the batch scripts by PRIORITY TIER
# (P0 first) and, within a tier, SLOW duration buckets first so the long-tail
# runs start early and never sit behind a queue of fast ones.
#
# Each batch script already packs a full GPU (16-24 runs via `&` ... `wait`), per
# the concurrency benchmark. So:
#   * single GPU (default): batches run one at a time, sequentially.
#   * multiple GPUs (--gpus 0,1,..): N batches run concurrently, one pinned per
#     GPU via CUDA_VISIBLE_DEVICES; the runner proceeds a "round" at a time.
#
# Because each batch fans out ~24 tiny-model processes onto ONE GPU, start MPS
# so their kernels share the SMs (the benchmark showed a large aggregate speedup):
#     ./run_ablation.sh --mps            # start a per-user MPS daemon first
#
# Usage:
#   ./run_ablation.sh                       # P0->P1->P2 on GPU 0, sequentially
#   ./run_ablation.sh --gpus 0,1,2,3        # spread batches across 4 GPUs
#   ./run_ablation.sh --mps --gpus 0,1      # + MPS
#   ./run_ablation.sh --tiers p0            # only the P0 tier
#   ./run_ablation.sh --include-eval        # also run P3 (needs filled checkpoints)
#   ./run_ablation.sh --list                # print the schedule and exit
#   ./run_ablation.sh --dry-run             # print the exact commands, run nothing
# ===========================================================================
set -uo pipefail

ABL_DIR="$(cd "$(dirname "$(realpath "$0")")" && pwd)"
REPO_ROOT="$(cd "${ABL_DIR}/../../../.." && pwd)"
MPS_HELPER="${REPO_ROOT}/benchmarks/mps_control.sh"

# ---- schedule: priority tier, then SLOW bucket first ----------------------
# P0: headline F1 + xl baseline, plus the independent diagnostics (D1 gradnorm,
#     D4 maskx). D1 has compile disabled -> slowest -> first.
P0=(
    p0_d1_8h4l.sh            # SLOW  (gradnorm, compile off, deep)
    p0_d1_1h1l.sh            # SLOW  (gradnorm, compile off)
    p0_f1_simple_trm.sh      # MEDIUM (4h3l)
    p0_f1_srm.sh             # MEDIUM (4h3l)
    p0_d4_maskx.sh           # MEDIUM (4h3l)
    p0_f1_mlp_xl.sh          # FAST   (no recursion)
)
# P1: remaining factorials.
P1=(
    p1_f3_slow.sh            # SLOW  (4h3l, 8h4l)
    p1_f6_8h4l.sh            # SLOW  (8h4l)
    p1_f5_reg.sh             # MEDIUM (4h3l)
    p1_f7_dis.sh             # MEDIUM (4h3l)
    p1_f4_hl.sh              # MEDIUM
    p1_f3_fast.sh            # FAST   (1h1l, 2h2l)
    p1_f6_1h1l.sh            # FAST   (1h1l)
)
# P2: 1-D sweeps + size baselines.
P2=(
    p2_s2_srm_trunc.sh       # SLOW-ish (deep recursion)
    p2_b2_model_size.sh      # MEDIUM
)
# P3: eval-only, DEPENDS on P0/P1 checkpoints -> off by default.
P3=(
    p3_d3_planning_cycles.sh
    p3_u1_video.sh
)

# ---- args -----------------------------------------------------------------
GPUS_CSV="0"
USE_MPS=0
DRY_RUN=0
LIST_ONLY=0
INCLUDE_EVAL=0
TIERS="p0 p1 p2"

while [ $# -gt 0 ]; do
    case "$1" in
        --gpus)          GPUS_CSV="$2"; shift 2 ;;
        --mps)           USE_MPS=1; shift ;;
        --dry-run)       DRY_RUN=1; shift ;;
        --list)          LIST_ONLY=1; shift ;;
        --include-eval)  INCLUDE_EVAL=1; TIERS="${TIERS} p3"; shift ;;
        --tiers)         TIERS="$2"; shift 2 ;;
        -h|--help)       sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

# ---- assemble the ordered batch list from the requested tiers -------------
BATCHES=()
for t in $TIERS; do
    case "$t" in
        p0) BATCHES+=("${P0[@]}") ;;
        p1) BATCHES+=("${P1[@]}") ;;
        p2) BATCHES+=("${P2[@]}") ;;
        p3) BATCHES+=("${P3[@]}") ;;
        *)  echo "unknown tier: $t"; exit 2 ;;
    esac
done

echo "==================================================================="
echo " ablation schedule (tiers: ${TIERS})"
echo " GPUs: ${GPUS[*]}   MPS: ${USE_MPS}   dry-run: ${DRY_RUN}"
echo "==================================================================="
n=0
for b in "${BATCHES[@]}"; do n=$((n+1)); printf '  %2d. %s\n' "$n" "$b"; done
echo "-------------------------------------------------------------------"
[ "$LIST_ONLY" -eq 1 ] && exit 0

# ---- MPS ------------------------------------------------------------------
if [ "$USE_MPS" -eq 1 ] && [ "$DRY_RUN" -eq 0 ]; then
    if [ -f "$MPS_HELPER" ]; then
        # shellcheck disable=SC1090
        source "$MPS_HELPER" start
    else
        echo "[warn] MPS helper not found at ${MPS_HELPER}; continuing without MPS."
    fi
fi

# ---- dispatch: one round = one batch per GPU ------------------------------
NG=${#GPUS[@]}
i=0
total=${#BATCHES[@]}
while [ "$i" -lt "$total" ]; do
    pids=()
    for g in "${GPUS[@]}"; do
        [ "$i" -ge "$total" ] && break
        batch="${ABL_DIR}/${BATCHES[$i]}"
        i=$((i+1))
        if [ ! -f "$batch" ]; then echo "[warn] missing batch: $batch"; continue; fi
        echo ">>> [GPU ${g}] $(basename "$batch")"
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "    CUDA_VISIBLE_DEVICES=${g} bash ${batch}"
        else
            CUDA_VISIBLE_DEVICES="${g}" bash "${batch}" &
            pids+=("$!")
        fi
    done
    # wait for this round's batches to finish before starting the next round
    if [ "$DRY_RUN" -eq 0 ]; then
        for p in "${pids[@]:-}"; do [ -n "${p:-}" ] && wait "$p"; done
    fi
done

echo "==================================================================="
echo " all requested ablation batches finished."
echo "==================================================================="
