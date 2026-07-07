# Newt benchmarks

Two independent benchmark scripts for the Newt / TD-MPC2 + SimpleTRM-dynamics
codebase. They share repo-inspection helpers (`newt_bench_common.py`) but have
separate entry points and separate JSON outputs, and **either runs on its own**.

| Script | Question | Entry point | Output |
|--------|----------|-------------|--------|
| **A** | How many training runs fit concurrently on the GPU, and what binds — GPU memory, GPU compute, or env/CPU? | `bench_concurrency.py` | `benchmark_results.json` |
| **B** | Where does a single run's wall-clock time go, per component and per recursion cycle? | `profile_components.py` | `profile_results.json` |

Both activate nothing on import; run them under the `newt` conda env from this
directory.

```bash
conda activate newt
cd benchmarks
python bench_concurrency.py            # Script A
python profile_components.py           # Script B
```

## What the scripts drive (repo inspection summary)

* **Entry point / launch.** `tdmpc2/train.py` → `parse_cfg()` → `Trainer` →
  `TDMPC2` → `WorldModel`. Runs are Hydra-configured; multi-GPU via
  `torch.multiprocessing.spawn` when `multiproc=True` (uses *all* visible GPUs),
  else single process. GPUs are selected with `CUDA_VISIBLE_DEVICES` +
  `torch.cuda.set_device(rank)`. Envs are a `gymnasium.AsyncVectorEnv` of
  `num_envs` **CPU subprocess workers** (default 21 for `dmcontrol`); the replay
  `Buffer` stores on `cuda:0`.
* **Why we don't shell out to `train.py`.** It burns
  `seeding_coef*num_envs*episode_length` (=52,500) pure env steps *before the
  first gradient step*, plus a step-0 eval, and has no `--max-steps` flag. Both
  scripts instead construct the **real objects** (`envs.make_env`, `WorldModel`,
  `TDMPC2.update()/.plan()`, `common.buffer.Buffer`) directly and seed the buffer
  with real random-action rollouts, so we control the step budget and teardown
  while exercising the identical code path.
* **Update path (as implemented).** `TDMPC2.update()`:
  `buffer.sample` → `encode(obs[1:])` (no-grad `next_z`) →
  `loss_fn` (`torch.compile`d, `reduce-overhead`): `encode(obs[0])`, a
  `horizon`-step latent rollout through `WorldModel.next` (the **SimpleTRM**
  recursive core, `H_cycles`×`L_cycles`), reward head, Q-ensemble, consistency
  MSE, SIGReg, optional DIS loss → `backward()` → grad-clip → `optim.step` →
  `soft_update_target_Q` → `update_pi`. Planning (`plan`/`_mppi`/`_estimate_value`,
  MPPI: `iterations`×`num_samples`, rolling the world model) is the **acting**
  path, profiled separately by Script B.
* **Representative config** (from `experiments/paper/maskx/*.sh`,
  `experiments/local_3seed.sh`): `task=dmcontrol num_envs=21 model_size=S
  use_trm_dynamics=simple obs=state`. Those scripts already launch **3 seeds
  concurrently on one GPU** (`... & ... & wait`) — exactly Script A's question.
* **SMALL vs LARGE.** SMALL = `latent_dim=16`, FiLM dynamics, no XL MLP. LARGE =
  `latent_dim=512` + XL dynamics MLP (`xl_dynamics_mlp` only affects the
  **non-FiLM** SimpleTRM path, so LARGE turns FiLM off). Both keep `model_size=S`
  and override `latent_dim`/cycles *after* `parse_cfg` (the `TRM_SIZE['S']` preset
  otherwise forces `latent_dim=384`/`4h3l`).

## Hardware detection

Both scripts print detected GPUs (count, VRAM, SM count, MIG status via
`nvidia-smi -L`), CPU core count, and system RAM. **No single-card / A100
assumption is baked in** — numbers are relative to whatever machine runs them.
(Developed/validated on 1× RTX 5080 16 GB, 32 cores, 64 GB RAM.)

## Script A — `bench_concurrency.py`

Sweeps concurrency `1,2,4,8,…` per config. Each run is a **separate subprocess**
pinned to a GPU, so an OOM or an init divergence is caught by the driver, not
fatal to the sweep, and every child is reaped (no orphans). Per level it reports
per-run peak GPU memory, env-build + first-reset wall time, steady env-steps/s
and grad-steps/s, aggregate GPU util / GPU mem / system CPU, and CPU
cores-equivalent per run, then attributes the bottleneck:

* **gpu_memory** — escalation hit a caught CUDA OOM.
* **gpu_compute** — aggregate grad-steps/s plateaus while GPU util stays high.
* **env_cpu** — throughput plateaus while GPU util is low and CPU is pegged.

Outputs a per-config table, a derived GPU-memory ceiling (runs/GPU from the
level-1 peak), a one-line runs-per-GPU scheduler recommendation, and
`benchmark_results.json` (per-config concurrency limits for a downstream
scheduler).

```bash
python bench_concurrency.py --config small,large --levels 1,2,4,8 --num-envs 21 --duration 8
python bench_concurrency.py --config small --num-envs 4        # fewer env workers -> pack more runs
python bench_concurrency.py --max-env-processes 200            # safety cap on total env workers
```

Key flags: `--levels`, `--max-concurrency`, `--num-envs`, `--duration`,
`--small-cycles/--large-cycles` (default `1h1l` — a numerically stable depth so
real grad steps run; see the divergence note below), `--compile` (match real runs).

## Script B — `profile_components.py`

Profiles a single run. **Timing method: CUDA events with `event.synchronize()`**
around every region (`common.CudaTimer` / `Timer`), so async kernels aren't
misattributed; `--no-cuda-timer` falls back to wall-clock + `torch.cuda.synchronize`.
Reports a **cold** pass and a **steady-state** mean over `--iters` warm iterations.

Components: env reset/step, encoder, dynamics single-step, dynamics rollout,
reward head, value/Q, policy pi, consistency MSE, planning/MPPI, and the full
update decomposed into **forward / backward / optimizer**. Because cost scales
with recursion depth, it also emits a **dynamics depth sweep** (`1h1l,1h2l,2h1l,…`)
and the **marginal ms per extra H-cycle and per extra L-cycle**. Cycle count is a
CLI parameter, and it runs SMALL and LARGE across e.g. `1h1l` and `8h4l`.

```bash
python profile_components.py                                   # SMALL+LARGE, 1h1l & 8h4l
python profile_components.py --config small --cycles 1h1l,4h3l,8h4l --warmup 20 --iters 50
python profile_components.py --config large --cycles 8h4l --no-cuda-timer
```

Timing hooks add ~µs/call and can be disabled via `--no-cuda-timer`.

## Findings that shaped the design (validated on the dev box)

1. **Params barely move VRAM.** LARGE has ~180× the dynamics params of SMALL
   (3.7M vs 20k) but peak GPU memory is ~473 MB vs ~378 MB — dominated by the
   CUDA context, replay buffer, and activations, *not* parameters. So the binding
   constraint on packing runs is throughput (env/CPU or GPU-compute
   serialization), not model size — matching the stated hypothesis. Script A is
   built to tell these apart rather than report a VRAM ceiling.
2. **Dynamics rollout dominates the forward** (~66–97 % of the isolated forward
   components), and **planning cost explodes with depth** (~16 ms at `1h1l` →
   ~112 ms at `8h4l`, since MPPI rolls the recursive core `iterations×horizon`
   times). This is the depth signal the compute-adaptivity analysis needs.
3. **The SwiGLU skip connection is quadratic and its `CastedLinear` weights use
   LeCun init that `weight_init` never touches**, so the `z` carry residual stream
   **overflows at fresh init** for deep recursion (`16→66→2154→…→inf` over
   `L_cycles`), worse at higher `L`. Both scripts handle this gracefully: Script B
   detects the non-finite loss and reports `diverged at init` (skipping the
   `backward` that would device-assert in `two_hot`); Script A isolates it in a
   subprocess and, via a finiteness-guarded update, records it as a divergence
   instead of crashing. The stable `1h1l` default lets real grad steps run for
   the throughput/memory numbers. **This is a genuine numerical-stability finding
   in the current `sigreg` branch, not a benchmark artifact.**
