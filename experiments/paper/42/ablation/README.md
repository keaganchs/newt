# TRM-dynamics ablation batch (42 style)

Experiment definitions for the dynamics ablation plan, in the same 42 convention
as `experiments/paper/42/latent_dim/*.sh` (no SBATCH; `MUJOCO_GL=disable`; the
`mlde_wsp_PI_Deramo` conda; runs packed on one GPU with `&` … `wait`; 3 seeds
`(0 1 2)` per config; `wandb_project="TRM Dynamics"` / `wandb_entity="trm-dynamics"`).

The one evolution vs. the hand-written 42 scripts: the anchor config and the
`python3 train.py …` block are factored into a sourced helper
(`_ablation_common.sh`) with a `launch` function, so the ~72 config cells stay
consistent instead of being copy-pasted. Same launch path, same hydra keys, same
seed handling and run naming — just DRY. Each cell is a **slice at the anchor**:
it overrides only the axis it sweeps and inherits everything else, so every
"off"/"baseline" cell is byte-for-byte the anchor.

## Anchor (held fixed except the swept axis)

```
use_trm_dynamics=simple  latent_dim=384  hidden_size=384
use_film_dynamics=False   wm_regularization_type=simnorm
use_dis_loss=False        xl_dynamics_mlp=False
use_simple_trm_skip_connections=False
rrm_mask_x_for_y_update=True
compile=True              log_trm_gradnorms=False
H_cycles=4  L_cycles=3     # the S-preset "recursion on" depth (see note 1)
task=dmcontrol  num_envs=21  obs=state  model_size=S  use_trm_encoder=False
```

## How to run

```bash
cd experiments/paper/42/ablation

./run_ablation.sh --list            # print the schedule, run nothing
./run_ablation.sh                   # P0 -> P1 -> P2, GPU 0, one batch at a time
./run_ablation.sh --mps --gpus 0    # start per-user MPS first (recommended: each
                                    #   batch fans ~24 tiny runs onto one GPU)
./run_ablation.sh --gpus 0,1,2,3    # spread batches across 4 GPUs (one per GPU)
./run_ablation.sh --tiers p0        # just the P0 tier
./run_ablation.sh --include-eval    # also P3 (needs checkpoints filled in, note 6)
```

The runner schedules by **priority tier (P0 first)** and, within a tier, **slow
duration buckets first** (so the long-tail runs start early). Each batch script
already packs a full GPU (16–24 runs), so the runner gives one batch per GPU and
proceeds a round at a time. You can also run any batch directly:
`CUDA_VISIBLE_DEVICES=0 bash p1_f5_reg.sh`.

## Batches, plan mapping, and groups

Every SET of runs has a unique `wandb_group` so its data can be pulled for a
plot; the cell (axis value + seed) is encoded in `wandb_run_name`.

| Tier | Batch | Plan | Cells×seeds | `wandb_group` | Bucket |
|------|-------|------|-------------|---------------|--------|
| P0 | `p0_f1_mlp_xl.sh` | F1 (MLP col) + F2 | 8×3=24 | `abl_f1_ld_x_dyn`, `abl_f2_mlp_xl` | fast |
| P0 | `p0_f1_simple_trm.sh` | F1 (simple, trm) | 8×3=24 | `abl_f1_ld_x_dyn` | med |
| P0 | `p0_f1_srm.sh` | F1 (srm) | 4×3=12 | `abl_f1_ld_x_dyn` | med |
| P0 | `p0_d1_1h1l.sh` | D1 @1h1l | 4×3=12 | `abl_d1_gradnorm` | slow¹ |
| P0 | `p0_d1_8h4l.sh` | D1 @8h4l | 4×3=12 | `abl_d1_gradnorm` | slowest¹ |
| P0 | `p0_d4_maskx.sh` | D4 | 2×3=6 | `abl_d4_maskx` | med |
| P1 | `p1_f3_fast.sh` | F3 (1h1l,2h2l) | 8×3=24 | `abl_f3_gate_x_cycles` | fast |
| P1 | `p1_f3_slow.sh` | F3 (4h3l,8h4l) | 8×3=24 | `abl_f3_gate_x_cycles` | slow |
| P1 | `p1_f4_hl.sh` | F4 | 3×3=9 | `abl_f4_h_vs_l` | med |
| P1 | `p1_f5_reg.sh` | F5 | 6×3=18 | `abl_f5_regularization` | med |
| P1 | `p1_f6_1h1l.sh` | F6 @1h1l | 2×3=6 | `abl_f6_film` | fast |
| P1 | `p1_f6_8h4l.sh` | F6 @8h4l | 2×3=6 | `abl_f6_film` | slow |
| P1 | `p1_f7_dis.sh` | F7 | 6×3=18 | `abl_f7_dis` | med |
| P2 | `p2_s2_srm_trunc.sh` | S2 | 5×3=15 | `abl_s2_srm_truncation` | slow |
| P2 | `p2_b2_model_size.sh` | B2 | 2×3=6 | `abl_b2_model_size` | med |
| P3 | `p3_d3_planning_cycles.sh` | D3 | eval-only | `abl_d3_planning_cycles` | fast |
| P3 | `p3_u1_video.sh` | U1 | eval-only | `abl_u1_video` | fast |

**Totals:** 72 training configs × 3 seeds = **216 training runs** across 15
batches, plus the two eval-only P3 templates. ¹ D1 sets `log_trm_gradnorms=True`,
which disables `torch.compile` on the recursive inner apply → much slower, so it
lives in its own batches (the only runs with gradnorm logging).

## Decisions & caveats (verify against `tdmpc2/config.py`)

1. **Anchor recursion depth = 4h3l.** The plan pins everything except cycles, so
   for experiments that don't sweep cycles (F1/F2/F5/F7/D4/S2) the anchor depth
   is the `S`-preset default `H=4,L=3`.
2. **`latent_dim` is always an explicit override**, so the `model_size`/`trm_size`
   presets don't clobber it. For the recursive variants (`simple`/`trm`/`srm`) we
   also set **`hidden_size=latent_dim`** (per the plan). The SimpleTRM anchor
   actually *ignores* `hidden_size` (its MLP path uses `latent_dim` internally,
   `simple_trm.py:48-53`); we set it so the `trm`/`srm` transformer width scales
   with the latent under one knob. `num_heads` is left at the `S` preset (**=2**),
   which evenly divides all four latent_dims (16/128/384/512 → even head_dim), and
   `simnorm_dim=8` divides them too — so no divisibility fixes are needed and the
   transformer path (active for `trm`/`srm`) is safe.
3. **`xl_dynamics_mlp` is combined with `use_trm_dynamics=None` only** (it's a
   no-op for simple/trm/srm). The F1 "compute-matched MLP baseline" IS the F2
   sweep: `none + xl` at all four latent_dims (F2's requested {16,384} is a
   subset; we run all four so the headline has a full MLP-XL column). The
   non-xl MLP baselines are the `abl_f1_ld_x_dyn` MLP cells.
4. **Gate/skip mechanism** (`use_simple_trm_skip_connections` + `simple_trm_skip_type`
   ∈ {additive, mlp, swiglu}) is SimpleTRM-specific, so the gate sweeps (D1, F3)
   run on `use_trm_dynamics=simple`. Gate "off" = `use_simple_trm_skip_connections=False`
   (skip_type irrelevant). S2 folds no skip-type comparison (per the plan).
5. **Mutually-exclusive regularizers**: `wm_regularization_type ∈ {simnorm, sigreg,
   none}` (F5). `sigreg` disables SimNorm and adds the SIGReg loss; `none` leaves
   the latent unconstrained. **DIS** (F7): `use_dis_loss=True` adds `dis_schedule
   ∈ {linear, cosine}` (swept only when on); off = exact anchor behaviour.
6. **P3 is eval-only and dependency-gated.** There is no separate eval entry point
   in this repo, so D3/U1 load a checkpoint and let the **step-0 evaluation** run,
   then exit (`checkpoint=… steps=1 save_agent=False`). They depend on trained
   F1/F3 checkpoints, so their `CHECKPOINTS` arrays are **empty by design** (the
   scripts abort until you fill them) and the runner **excludes P3 unless
   `--include-eval`**. Each entry carries the arch overrides needed to rebuild the
   model identically before loading weights. D3 sweeps `planning_H_cycles ∈
   {1,2,4,8}` using the planning-depth override (decouples MPPI-rollout depth from
   trained depth); add `planning_L_cycles` there if you want the L axis too.
7. **B2** is the Newt MLP baseline at `model_size ∈ {M, L}` (matches the existing
   `newt_m_l` scripts: `use_trm_dynamics=False`, `latent_dim=128`). It uses
   `launch_raw` so the size preset — not the recursive anchor — sets
   `enc_dim`/`mlp_dim`/`num_q`/`num_enc_layers`.
8. **Metrics.** `num_dynamics_params` is filled at init and logged to the run
   config automatically; wall-clock throughput is logged as `steps_per_second`
   (trainer). **FLOPs are not auto-logged** — use `benchmarks/profile_components.py`
   for per-cycle time/FLOP marginals (the F1 param/FLOP/wall-clock table combines
   the WandB config fields with a profiler pass).
9. **D2** (per-cycle Advantage Margin logging) is out of scope for this batch, as
   the plan states.
