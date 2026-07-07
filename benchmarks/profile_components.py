#!/usr/bin/env python
"""
Script B -- Per-component time profiler for a SINGLE Newt training run.

Reports where wall-clock time goes in one run, in absolute (real) ms and as a
percentage of total, broken down into the actual TD-MPC2/Newt components as
implemented in this repo:

    env.reset / env.step        (CPU async workers)
    encoder                     WorldModel.encode  -> self._encoder
    dynamics rollout            WorldModel.next    -> self._dynamics (SimpleTRM)
       + per-H-cycle / per-L-cycle marginal cost (recursion depth sweep)
    reward head                 WorldModel.reward  -> self._reward
    value / Q                   WorldModel.Q       -> self._Qs
    policy (pi)                 WorldModel.pi      -> self._pi
    consistency loss            F.mse_loss in TDMPC2._loss_fn
    planning (MPPI)             TDMPC2.plan / _mppi / _estimate_value
    update forward / backward / optimizer step   (TDMPC2.update decomposed)

Timing method: CUDA events with event.synchronize() around every measured
region (common.CudaTimer), so GPU async execution is not misattributed. We
report a COLD pass (first call: includes torch.compile / cuDNN autotune / first
CUDA kernels) and a STEADY-STATE average over N warm iterations.

The per-component microbenchmarks call the REAL modules in isolation, which is
both more precise (no torch.compile fusion hiding component boundaries) and
robust: the discrete-regression two_hot targets are built from real rewards and
encode(next_z) (always finite), never from the dynamics rollout, so a deep
recursion that overflows at init does not trigger a device-side assert here.

Runs across SMALL (latent_dim=16) and LARGE (latent_dim=512 + XL dynamics) and
across cycle counts (default 1h1l and 8h4l) so parameter- and depth-dependence
are both visible. Cycle count is a CLI parameter.

Self-contained; timing hooks add ~microseconds/call and can be disabled with
--no-cuda-timer (falls back to wall-clock with explicit synchronize()).

Usage:
    python profile_components.py                       # SMALL+LARGE, 1h1l & 8h4l
    python profile_components.py --config small --cycles 8h4l --warmup 20 --iters 50
    python profile_components.py --config large --cycles 1h1l,4h3l,8h4l
"""
import os
import sys
import json
import time
import argparse
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import newt_bench_common as C


def parse_cycles(s):
    """'8h4l' -> (8, 4); '1h1l,4h3l' -> [(1,1),(4,3)]."""
    out = []
    for tok in s.split(","):
        tok = tok.strip().lower()
        h, l = tok.split("h")
        l = l.rstrip("l")
        out.append((int(h), int(l)))
    return out


def spec_for(config, h, l, num_envs):
    if config == "small":
        return C.small_spec(h=h, l=l, num_envs=num_envs)
    elif config == "large":
        return C.large_spec(h=h, l=l, num_envs=num_envs)
    raise ValueError(config)


class Timer:
    """Unifies CUDA-event timing (default) and wall-clock timing (--no-cuda-timer)."""
    def __init__(self, use_cuda_events=True):
        import torch
        self.torch = torch
        self.use_cuda = use_cuda_events and torch.cuda.is_available()

    @contextlib.contextmanager
    def measure(self):
        torch = self.torch
        if self.use_cuda:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            box = {}
            yield box
            e.record()
            e.synchronize()
            box["ms"] = s.elapsed_time(e)
        else:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            box = {}
            yield box
            torch.cuda.synchronize()
            box["ms"] = (time.perf_counter() - t0) * 1e3

    def bench(self, fn, warmup, iters):
        """Returns (cold_ms, steady_mean_ms, steady_std_ms)."""
        import statistics
        # cold
        with self.measure() as b:
            fn()
        cold = b["ms"]
        for _ in range(max(0, warmup - 1)):
            fn()
        samples = []
        for _ in range(iters):
            with self.measure() as b:
                fn()
            samples.append(b["ms"])
        mean = statistics.mean(samples) if samples else float("nan")
        std = statistics.pstdev(samples) if len(samples) > 1 else 0.0
        return cold, mean, std


def profile_one(config, h, l, num_envs, warmup, iters, use_cuda_events, compile_full):
    import torch
    import torch.nn.functional as F

    tag = f"{config.upper()} {h}h{l}l"
    print(f"\n{'='*70}\n  PROFILING {tag}  (num_envs={num_envs}, warmup={warmup}, iters={iters})\n{'='*70}")
    spec = spec_for(config, h, l, num_envs)

    # Component microbenchmarks want compile OFF (so boundaries are real, not fused).
    cfg = C.build_cfg(spec, compile=False, log_trm_gradnorms=False)
    env = C.build_env(cfg)
    agent = C.build_agent(cfg)
    buf = C.build_buffer(cfg, capacity=200_000)

    timer = Timer(use_cuda_events)
    result = {
        "config": config, "H_cycles": h, "L_cycles": l, "num_envs": num_envs,
        "latent_dim": cfg.latent_dim, "task_dim": cfg.task_dim, "action_dim": cfg.action_dim,
        "batch_size": cfg.batch_size, "horizon": cfg.horizon,
        "use_film_dynamics": cfg.use_film_dynamics, "xl_dynamics_mlp": cfg.xl_dynamics_mlp,
        "dyn_params": cfg.num_dynamics_params, "total_params": agent.model.total_params,
        "warmup": warmup, "iters": iters,
        "timing_method": "cuda_events" if timer.use_cuda else "wallclock_sync",
        "components": {},   # name -> {cold_ms, steady_ms, std_ms}
    }

    # ---- env timing (reset + step), before we seed ------------------------ #
    with timer.measure() as b:
        obs, info = env.reset()
    result["components"]["env_reset"] = {"cold_ms": b["ms"], "steady_ms": None, "std_ms": None}

    N = cfg.num_envs
    tasks = torch.arange(N, dtype=torch.int32)

    def env_step():
        a = env.rand_act()
        env.step(a)
    cold, mean, std = timer.bench(env_step, warmup=min(5, warmup), iters=min(iters, 30))
    # env.step returns per-env-batch; normalise to per-1-env-step for comparability
    result["components"]["env_step (all %d envs)" % N] = {"cold_ms": cold, "steady_ms": mean, "std_ms": std}
    result["env_steps_per_sec"] = round(N / (mean / 1e3), 1) if mean and mean > 0 else None

    # ---- seed the buffer with real rollouts ------------------------------- #
    t0 = time.time()
    C.seed_buffer_with_rollouts(env, cfg, buf, num_episodes_worth=1)
    result["seed_wall_s"] = round(time.time() - t0, 2)

    # A representative training batch (real distribution).
    obs, action, reward, task = buf.sample(device=agent.device)
    B = cfg.batch_size
    agent.model.train()

    # ---- isolated component microbenchmarks ------------------------------- #
    z0 = agent.model.encode(obs[0], task[0]).detach()          # [B, ld]
    a0 = action[0]                                             # [B, act]
    t1 = task[0]

    def do_encode():
        with torch.no_grad():
            agent.model.encode(obs[0], task[0])

    def do_dyn_step():
        with torch.no_grad():
            r = agent.model.next(z0, a0, t1)
            return r

    def do_rollout():
        with torch.no_grad():
            z = z0
            for tt in range(cfg.horizon):
                z = agent.model.next(z, action[tt], task[tt])
                if isinstance(z, tuple):
                    z = z[0]

    _zs = torch.stack([z0] * cfg.horizon)                     # [H, B, ld]

    def do_reward():
        with torch.no_grad():
            agent.model.reward(_zs, action, task)

    def do_Q_all():
        with torch.no_grad():
            agent.model.Q(_zs, action, task, return_type="all")

    def do_pi():
        with torch.no_grad():
            agent.model.pi(_zs, task)

    zt = torch.stack([z0] * (cfg.horizon + 1))
    zt2 = torch.roll(zt, 1, 0)

    def do_consistency():
        F.mse_loss(zt, zt2)

    comps = [
        ("encoder", do_encode),
        ("dynamics_step (1x)", do_dyn_step),
        ("dynamics_rollout (%dx)" % cfg.horizon, do_rollout),
        ("reward_head", do_reward),
        ("value_Q (all)", do_Q_all),
        ("policy_pi", do_pi),
        ("consistency_mse", do_consistency),
    ]
    for name, fn in comps:
        try:
            cold, mean, std = timer.bench(fn, warmup, iters)
            result["components"][name] = {"cold_ms": round(cold, 4), "steady_ms": round(mean, 4), "std_ms": round(std, 4)}
        except Exception as ex:
            result["components"][name] = {"error": f"{type(ex).__name__}: {ex}"}

    # ---- planning (MPPI) full path ---------------------------------------- #
    # plan() expects a single obs batch of shape [num_envs, 128] and t0 flags.
    plan_obs = obs[0][:N].contiguous()                        # [N, 128]
    plan_task = tasks.to(agent.device)
    t0flag = torch.ones(N, dtype=torch.bool, device=agent.device)
    agent.model.eval()

    def do_plan():
        with torch.no_grad():
            agent.plan(plan_obs, t0=t0flag, step=10_000_000, eval_mode=True, task=plan_task)
    try:
        cold, mean, std = timer.bench(do_plan, min(warmup, 5), min(iters, 20))
        result["components"]["planning_mppi (%d iters x %d samples)" % (cfg.iterations, cfg.num_samples)] = \
            {"cold_ms": round(cold, 4), "steady_ms": round(mean, 4), "std_ms": round(std, 4)}
    except Exception as ex:
        result["components"]["planning_mppi"] = {"error": f"{type(ex).__name__}: {ex}"}
    agent.model.train()

    # ---- per-cycle dynamics depth sweep ----------------------------------- #
    # Marginal cost of an extra recursion cycle: time the dynamics rollout while
    # sweeping H and L, so d(time)/d(cycle) is directly visible. This is the knob
    # the compute-adaptivity analysis cares about.
    depth_sweep = {}
    base_H, base_L = cfg.H_cycles, cfg.L_cycles
    sweep_points = sorted(set([(1, 1), (1, 2), (2, 1), (base_H, base_L),
                               (base_H, max(1, base_L + 1)), (max(1, base_H + 1), base_L)]))
    for (hh, ll) in sweep_points:
        agent.model._dynamics.config.H_cycles = hh
        agent.model._dynamics.config.L_cycles = ll
        def do_dyn():
            with torch.no_grad():
                r = agent.model.next(z0, a0, t1)
                return r
        try:
            _, mean, _ = timer.bench(do_dyn, min(warmup, 10), min(iters, 40))
            depth_sweep[f"{hh}h{ll}l"] = round(mean, 4)
        except Exception as ex:
            depth_sweep[f"{hh}h{ll}l"] = None
    agent.model._dynamics.config.H_cycles = base_H
    agent.model._dynamics.config.L_cycles = base_L
    result["dynamics_depth_sweep_ms"] = depth_sweep
    # marginal per-L and per-H (using base point vs +1)
    def _get(k):
        v = depth_sweep.get(k)
        return v
    if _get(f"{base_H}h{base_L}l") is not None and _get(f"{base_H}h{base_L+1}l") is not None:
        result["marginal_per_L_cycle_ms"] = round(_get(f"{base_H}h{base_L+1}l") - _get(f"{base_H}h{base_L}l"), 4)
    if _get(f"{base_H}h{base_L}l") is not None and _get(f"{base_H+1}h{base_L}l") is not None:
        result["marginal_per_H_cycle_ms"] = round(_get(f"{base_H+1}h{base_L}l") - _get(f"{base_H}h{base_L}l"), 4)

    # ---- full update: forward / backward / optimizer split ---------------- #
    # Decompose TDMPC2.update()'s core exactly. Guarded: if the config overflows
    # at init (deep swiglu-skip recursion), we detect the non-finite loss and skip
    # backward (which is where the two_hot scatter would device-assert), reporting
    # the divergence rather than crashing the whole profiler.
    fb = {"note": None}
    try:
        # a couple of warmup updates via the real update() to reach steady state
        n_warm_ok = 0
        for _ in range(min(warmup, 8)):
            with torch.no_grad():
                next_z = agent.model.encode(obs[1:], task)
            if not torch.isfinite(next_z).all():
                fb["note"] = "encode(next_z) non-finite"
                break
            torch.compiler.cudagraph_mark_step_begin()
            total_loss, zs, info = agent.loss_fn(obs, action, reward, next_z, task)
            if not torch.isfinite(total_loss):
                fb["note"] = f"loss non-finite at warmup (diverged at init: {info.get('consistency_loss')})"
                break
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), cfg.grad_clip_norm)
            agent.optim.step()
            agent.optim.zero_grad(set_to_none=True)
            n_warm_ok += 1

        if fb["note"] is None:
            fwd_ms, bwd_ms, opt_ms = [], [], []
            for _ in range(iters):
                with torch.no_grad():
                    next_z = agent.model.encode(obs[1:], task)
                torch.compiler.cudagraph_mark_step_begin()
                with timer.measure() as b:
                    total_loss, zs, info = agent.loss_fn(obs, action, reward, next_z, task)
                fwd_ms.append(b["ms"])
                with timer.measure() as b:
                    total_loss.backward()
                bwd_ms.append(b["ms"])
                with timer.measure() as b:
                    torch.nn.utils.clip_grad_norm_(agent.model.parameters(), cfg.grad_clip_norm)
                    agent.optim.step()
                    agent.optim.zero_grad(set_to_none=True)
                opt_ms.append(b["ms"])
            import statistics
            fb["update_forward_ms"] = round(statistics.mean(fwd_ms), 4)
            fb["update_backward_ms"] = round(statistics.mean(bwd_ms), 4)
            fb["update_optimizer_ms"] = round(statistics.mean(opt_ms), 4)
            fb["update_total_ms"] = round(fb["update_forward_ms"] + fb["update_backward_ms"] + fb["update_optimizer_ms"], 4)
            fb["warm_updates_ok"] = n_warm_ok
    except Exception as ex:
        fb["note"] = f"crashed: {type(ex).__name__}: {ex}"
    result["update_fwd_bwd_opt"] = fb

    result["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    try:
        env.close()
    except Exception:
        pass
    del agent, buf, env
    torch.cuda.empty_cache()
    return result


def print_table(result):
    comps = result["components"]
    # Build a single "update" attribution: encoder + rollout + reward + Q + pi + consistency
    # (isolated steady-state numbers) plus the measured fwd/bwd/opt if available.
    print(f"\n  Component steady-state times (ms) for {result['config'].upper()} "
          f"{result['H_cycles']}h{result['L_cycles']}l  "
          f"[dyn_params={result['dyn_params']:,}, total={result['total_params']:,}]")
    rows = []
    for name, d in comps.items():
        if "error" in d:
            rows.append((name, None, d["error"]))
        else:
            rows.append((name, d.get("steady_ms"), d.get("cold_ms")))
    # attribute % over the isolated "update-ish" components (exclude env/plan)
    update_names = ["encoder", "dynamics_rollout", "reward_head", "value_Q", "policy_pi", "consistency_mse"]
    tot = 0.0
    for name, d in comps.items():
        if any(name.startswith(u) for u in update_names) and isinstance(d.get("steady_ms"), (int, float)):
            tot += d["steady_ms"]
    print(f"  {'component':<40} {'steady ms':>12} {'cold ms':>12} {'% of fwd-sum':>14}")
    print("  " + "-" * 80)
    for name, d in comps.items():
        s = d.get("steady_ms") if "error" not in d else None
        c = d.get("cold_ms") if "error" not in d else d["error"]
        pct = ""
        if any(name.startswith(u) for u in update_names) and isinstance(s, (int, float)) and tot > 0:
            pct = f"{100*s/tot:6.1f}%"
        s_str = f"{s:12.4f}" if isinstance(s, (int, float)) else f"{'--':>12}"
        c_str = f"{c:12.4f}" if isinstance(c, (int, float)) else f"{str(c):>12}"
        print(f"  {name:<40} {s_str} {c_str} {pct:>14}")
    print("  " + "-" * 80)
    ds = result.get("dynamics_depth_sweep_ms", {})
    print(f"  dynamics depth sweep (ms, isolated 1x call): {ds}")
    if "marginal_per_L_cycle_ms" in result:
        print(f"    marginal per +1 L-cycle: {result['marginal_per_L_cycle_ms']} ms")
    if "marginal_per_H_cycle_ms" in result:
        print(f"    marginal per +1 H-cycle: {result['marginal_per_H_cycle_ms']} ms")
    fb = result.get("update_fwd_bwd_opt", {})
    if fb.get("note"):
        print(f"  full update fwd/bwd/opt: NOT MEASURED -> {fb['note']}")
    else:
        tt = fb.get("update_total_ms")
        print(f"  full update decomposition (compile OFF):")
        for k in ["update_forward_ms", "update_backward_ms", "update_optimizer_ms"]:
            v = fb.get(k)
            if v is not None and tt:
                print(f"    {k:<24} {v:10.4f} ms  ({100*v/tt:5.1f}%)")
        print(f"    {'update_total':<24} {tt:10.4f} ms")
    print(f"  peak GPU mem: {result.get('peak_gpu_mem_mb')} MB   "
          f"env_step throughput: {result.get('env_steps_per_sec')} env-steps/s")


def main():
    ap = argparse.ArgumentParser(description="Newt per-component time profiler (Script B).")
    ap.add_argument("--config", default="small,large", help="comma list of: small, large")
    ap.add_argument("--cycles", default="1h1l,8h4l", help="comma list like 1h1l,4h3l,8h4l")
    ap.add_argument("--num-envs", type=int, default=8, help="env workers for env-step timing")
    ap.add_argument("--warmup", type=int, default=15, help="warm iterations before steady-state timing")
    ap.add_argument("--iters", type=int, default=40, help="steady-state iterations to average")
    ap.add_argument("--no-cuda-timer", action="store_true", help="use wall-clock+sync instead of CUDA events")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_results.json"))
    args = ap.parse_args()

    hw = C.detect_hardware()
    print("Hardware:", json.dumps(hw, indent=2))
    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    cycles = parse_cycles(args.cycles)

    all_results = {"hardware": hw, "runs": []}
    for config in configs:
        for (h, l) in cycles:
            try:
                res = profile_one(config, h, l, args.num_envs, args.warmup, args.iters,
                                  not args.no_cuda_timer, compile_full=False)
                print_table(res)
                all_results["runs"].append(res)
            except Exception as ex:
                import traceback
                traceback.print_exc()
                all_results["runs"].append({"config": config, "H_cycles": h, "L_cycles": l,
                                            "fatal_error": f"{type(ex).__name__}: {ex}"})
            # write incrementally so a later crash never loses earlier results
            with open(args.out, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
