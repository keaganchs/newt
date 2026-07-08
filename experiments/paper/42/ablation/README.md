# TRM-dynamics ablation — axis-folder layout, two clusters

Experiment definitions for the dynamics ablation plan, organised **one folder per
sweep axis** and **split across two clusters**. Filenames encode the **architecture**
(`newt` = MLP dynamics, `smp` = SimpleTRM, `trm` = TRM, `srm` = SRM) and, where it
varies, the recursion depth (`1h1l`, `2h2l`, `4h3l`, `8h4l`).

## Two clusters, two launch styles

| | **A100 cluster** (no time limit) | **3090 cluster** (3 GPUs, 3-day limit) |
|---|---|---|
| Files | `*.sh` (42-style) | `*.slurm.sh` (SBATCH) |
| Launch | one script per shell, **one GPU visible**; `bash <axis>/<script>.sh` | `sbatch <axis>/<script>.slurm.sh` **from repo root** |
| Conda / paths | hardcoded `pfss/…mlde_wsp_PI_Deramo`, `MUJOCO_GL=disable` | `~/miniconda3`, `$SLURM_SUBMIT_DIR`, `--partition=main` |
| MPS | **on by default** (`--disable-mps` / `DISABLE_MPS=1` to turn off) | none (1 cell / 3 seeds per job) |
| Gets | everything medium/deep + logging + eval | **low-recursion only** (`1h1l`, `2h2l`) |

**Why this split:** the 3090s are faster but time-limited, so they take the fast
low-recursion cells that finish well inside 3 days. The A100s (unlimited) take the
`4h3l`/`8h4l` runs, the gradnorm-logging runs, and the eval-only runs.

Each A100 script is sized to fill one GPU (≈12–24 runs) and is meant to be started
in its own single-GPU shell (the shell sets `CUDA_VISIBLE_DEVICES`; the scripts do
not pin GPUs). With 12 A100s you can run 12 A100 scripts at once.

## Anchor (held fixed except the swept axis)

```
use_trm_dynamics=simple  latent_dim=384  hidden_size=384
use_film_dynamics=False   wm_regularization_type=simnorm
use_dis_loss=False        xl_dynamics_mlp=False
use_simple_trm_skip_connections=False
rrm_mask_x_for_y_update=True
compile=True              log_trm_gradnorms=False
H_cycles=4  L_cycles=3
task=dmcontrol  num_envs=21  obs=state  model_size=S  use_trm_encoder=False
```

The A100 anchor + `launch` helper live in `_ablation_common.sh`; the SBATCH files
are self-contained (they can't source under `sbatch`), generated from the same anchor.

## W&B grouping

`wandb_group` is set **per cell** as `<plan_tag>/<run_name>` (e.g.
`abl_latent_dim/smp_384ld_4h3l`). The 3 seeds of a cell share the group, so *group
by group* aggregates them into one mean±band line, while every cell is a distinct
group. To pull a whole plan, filter groups by the `<plan_tag>/` prefix.

## Axes → plan → cluster

| Folder (axis) | Plan | `plan_tag` | Cluster | Cells×seeds |
|---|---|---|---|---|
| `latent_dim/` | F1 headline (newt/smp/trm/srm × {16,128,384,512} @4h3l) | `abl_latent_dim` | A100 | 20×3 |
| `latent_dim/` | F2 compute-matched MLP+xl × ld | `abl_latent_dim_xl` | A100 | 4×3 |
| `latent_dim/` | low-recursion latent sweep (smp × ld @1h1l) | `abl_latent_dim_1h1l` | 3090 | 4×3 |
| `cycles/` | F3 gate {off,additive,mlp,swiglu} × {1h1l,2h2l,4h3l,8h4l} | `abl_cycles_gate` | split¹ | 16×3 |
| `h_vs_l/` | F4 (H,L) ∈ {(4,1),(2,2),(1,4)} | `abl_h_vs_l` | A100 | 3×3 |
| `regularization/` | F5 {simnorm,sigreg,none} × ld {16,384} | `abl_regularization` | A100 | 6×3 |
| `film/` | F6 film {False,True} × {1h1l,8h4l} | `abl_film` | split¹ | 4×3 |
| `dis_loss/` | F7 dis off/on × ld {16,384} (+ schedule) | `abl_dis_loss` | A100 | 6×3 |
| `srm_truncation/` | S2 truncation {1,2,3,6,12} @4h3l | `abl_srm_truncation` | A100 | 5×3 |
| `model_size/` | B2 Newt MLP {M,L} | `abl_model_size` | A100 | 2×3 |
| `mask_x/` | D4 rrm_mask_x_for_y_update {True,False} | `abl_mask_x` | A100 | 2×3 |
| `gradnorm/` | D1 gradnorm logging, gate × {1h1l,8h4l} | `abl_gradnorm` | A100 | 8×3 |
| `planning_cycles/` | D3 planning_H_cycles {1,2,4,8} | `abl_d3_planning_cycles` | A100, eval | — |
| `video/` | U1 rollout videos | `abl_u1_video` | A100, eval | — |
| `bench/` | 3090 concurrency benchmark | — | 3090 | — |

¹ **split**: `1h1l`/`2h2l` cells are `*.slurm.sh` (3090); `4h3l`/`8h4l` are `*.sh` (A100).

**Totals:** 76 training cells × 3 seeds = **228 runs** (186 on A100, 42 on 3090),
plus the two eval-only templates and the benchmark.

## Crash handling — 8h4l runs are isolated

Two `smp / 8h4l / swiglu-skip` runs crashed mid-training with
`dm_control … PhysicsError: mjWARN_BADCTRL` (the model emitted invalid controls).
To keep this from wasting a shared GPU, **every `8h4l` cell is its own script**
(`cycles/smp_gate_*_8h4l.sh`, `film/smp_*_8h4l.sh`, `gradnorm/smp_gate_*_8h4l.sh`) —
3 seeds each. A crash there ends only that script and frees its GPU cleanly instead
of leaving a packed GPU under-utilised. Root-causing the instability (action
clamping / grad-norm / lr for deep swiglu-skip) is deferred — see the plan.

## How to run

**A100 (per shell, one GPU each):**
```bash
cd <repo root>/experiments/paper/42/ablation
bash latent_dim/smp_ld.sh                 # MPS on by default
bash cycles/smp_gate_swiglu_8h4l.sh       # an isolated deep run
bash regularization/smp_reg.sh --disable-mps   # opt out of MPS
```
Start one script per available A100 shell. Suggested priority: `latent_dim/*`
(headline) and `gradnorm/*_8h4l` (slowest — gradnorm hooks graph-break the
recursion to eager, though the rest still compiles) first; then the rest.

**3090 (SBATCH, from repo root):**
```bash
cd <repo root>
sbatch experiments/paper/42/ablation/bench/bench_3090_concurrency.slurm.sh   # first: how many fit?
for f in experiments/paper/42/ablation/**/*.slurm.sh; do sbatch "$f"; done    # or submit individually
```
Each SBATCH job requests `--partition=main`, 1 GPU, 9 cpus, 5500 MB/cpu, 71:59:00,
and runs 1 cell × 3 seeds (what you found fits). If the benchmark shows >3 runs fit,
raise the per-job cell count.

## Eval-only (P3) — dependency-gated

`planning_cycles/eval_smp_d3.sh` and `video/eval_u1.sh` load **trained F1/F3
checkpoints** and run the step-0 eval only (`checkpoint=… steps=1 save_agent=False`).
Their `CHECKPOINTS` arrays are **empty by design** (the scripts abort until filled).
`save_video=True` in U1 forces `env_mode=sync`.

## Notes

- **`latent_dim` is always an explicit override** so the `model_size`/`trm_size`
  presets don't clobber it; for recursive variants `hidden_size=latent_dim` too.
  `num_heads` stays at the `S` preset (=2), which divides every latent_dim (16/128/
  384/512), and `simnorm_dim=8` divides them too — no divisibility fixes needed.
- **`xl_dynamics_mlp` only with `use_trm_dynamics=None`** (no-op for simple/trm/srm).
- **B2** uses `launch_raw` so the size preset sets `enc_dim`/`mlp_dim`/`num_q`.
- **Gradnorm runs keep `compile=True`.** `log_trm_gradnorms` doesn't need a global
  `compile=False`: `SimpleTRM.forward` is absorbed into the `loss_fn` compile
  (`simple_trm.py:123`), and the gradnorm `register_hook`s (`simple_trm.py:208,221`)
  only graph-break the recursive region to eager — the encoder/reward/Q/policy/MPPI
  still compile. This matches the original gradnorm runs (e.g. `latent_dim/384ld.sh`).
- **Partition update:** the non-42 SLURM scripts (`paper/sigreg`, `paper/srm`,
  `paper/gradnorm`, …) were moved from `--partition=amd2` to `--partition=main`
  (cluster update, not an access change).
- **Metrics:** `num_dynamics_params` and `steps_per_second` are auto-logged; FLOPs
  need `benchmarks/profile_components.py`.
