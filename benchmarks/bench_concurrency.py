#!/usr/bin/env python
"""
Script A -- Concurrency benchmark: how many Newt training runs fit at once, and
WHAT the binding constraint is (GPU memory, GPU compute, or env/CPU throughput).

Motivation (from the repo's own experiment scripts, e.g. experiments/local_3seed.sh
and experiments/paper/maskx/*.sh): those launch 3 seeds concurrently on one GPU
(`python train.py ... & ... & wait`). This script generalises that question and,
crucially, distinguishes the three possible bottlenecks -- because for this model
the parameters are tiny (SMALL dyn=20k params) while each run also spins up 21
async MuJoCo env workers, so CPU/env throughput can bind long before GPU memory.

Design
------
* Each run is a separate SUBPROCESS (`--worker`), pinned to a GPU via
  CUDA_VISIBLE_DEVICES. Subprocess isolation means an OOM or a divergence in one
  run is caught by the driver instead of taking down the sweep, and there are no
  orphaned env/GPU processes (the driver reaps every child).
* The driver sweeps concurrency 1,2,4,8,... per config, launching that many
  workers simultaneously, monitoring aggregate GPU util / GPU mem / system CPU
  while they run, then aggregating per-run throughput.
* Each worker drives the REAL objects (envs.make_env, WorldModel, TDMPC2.update,
  common.buffer.Buffer) in a short combined collect+update loop at the real UTD
  ratio, and reports: env-build + first-reset wall time, steady env-steps/sec and
  grad-steps/sec, per-run peak GPU memory, and CPU cores-equivalent + RSS for its
  whole process tree (worker + env workers).

Bottleneck attribution per config:
  - memory  : escalation hit CUDA OOM (caught) -> memory-max concurrency found.
  - compute : aggregate grad-steps/sec plateaus while GPU utilisation stays high.
  - env/cpu : aggregate throughput plateaus while GPU util is LOW and CPU pegged.

Two representative configs (grounded on the maskx scripts): SMALL (latent_dim=16)
and LARGE (latent_dim=512 + XL dynamics), since VRAM is parameter-dependent.

Constraints honoured: probes are short; CUDA is freed and processes reaped between
levels; OOM is caught/logged not crashed; no orphaned processes.

Usage:
    python bench_concurrency.py                          # SMALL+LARGE sweep
    python bench_concurrency.py --config small --levels 1,2,4,8 --duration 8
    python bench_concurrency.py --num-envs 21 --max-concurrency 8
    # (internal) python bench_concurrency.py --worker --config small ...
"""
import os
import sys
import json
import time
import argparse
import subprocess
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# WORKER  (one training run, its own process, pinned to one GPU)
# ===========================================================================
def run_worker(args):
    import newt_bench_common as C
    import torch

    out = {"config": args.config, "cycles": f"{args.h}h{args.l}l", "num_envs": args.num_envs,
           "pid": os.getpid(), "ok": False, "diverged": False, "error": None, "mode": "grad"}
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None

    try:
        if args.config == "small":
            spec = C.small_spec(h=args.h, l=args.l, num_envs=args.num_envs)
        else:
            spec = C.large_spec(h=args.h, l=args.l, num_envs=args.num_envs)
        cfg = C.build_cfg(spec, compile=args.compile)

        t0 = time.time()
        env = C.build_env(cfg)
        out["env_build_s"] = round(time.time() - t0, 3)
        agent = C.build_agent(cfg)
        buf = C.build_buffer(cfg, capacity=args.buffer_capacity)
        out["dyn_params"] = cfg.num_dynamics_params
        out["total_params"] = agent.model.total_params

        t0 = time.time()
        obs, info = env.reset()
        out["first_reset_s"] = round(time.time() - t0, 3)

        # Seed buffer with one episode-worth of real rollouts so update() has data.
        t0 = time.time()
        C.seed_buffer_with_rollouts(env, cfg, buf, num_episodes_worth=1)
        out["seed_s"] = round(time.time() - t0, 2)

        rho = agent.rho
        N = cfg.num_envs
        tasks = torch.arange(N, dtype=torch.int32)

        # ---- finiteness-guarded update (mirrors TDMPC2.update core) --------
        # Avoids the two_hot device-assert if a deep swiglu-skip config overflows
        # at init: on a non-finite loss we skip backward and flag divergence.
        def safe_update():
            o, a, r, tk = buf.sample(device=agent.device)
            agent.model.train()
            with torch.no_grad():
                next_z = agent.model.encode(o[1:], tk)
            if not torch.isfinite(next_z).all():
                return False
            torch.compiler.cudagraph_mark_step_begin()
            total_loss, zs, info = agent.loss_fn(o, a, r, next_z, tk)
            if not torch.isfinite(total_loss):
                return False
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), cfg.grad_clip_norm)
            agent.optim.step()
            agent.optim.zero_grad(set_to_none=True)
            agent.model.soft_update_target_Q()
            agent.update_pi(zs, a, tk[:1])
            agent.model.eval()
            return True

        # ---- warmup --------------------------------------------------------
        done = torch.ones(N, dtype=torch.bool)
        for _ in range(3):
            action = env.rand_act()
            env.step(action)
        warm_ok = 0
        for _ in range(args.warmup):
            if safe_update():
                warm_ok += 1
        if warm_ok == 0:
            out["diverged"] = True
            out["mode"] = "forward_only"

        # ---- steady-state combined collect+update loop ---------------------
        torch.cuda.reset_peak_memory_stats()
        if proc is not None:
            proc.cpu_percent(None)
            for ch in proc.children(recursive=True):
                try:
                    ch.cpu_percent(None)
                except Exception:
                    pass
        env_steps = 0
        grad_steps = 0
        update_tokens = 0.0
        t_start = time.time()
        t_end = t_start + args.duration
        while time.time() < t_end:
            action = env.rand_act()
            obs, reward, term, trunc, info = env.step(action)
            env_steps += N
            update_tokens += N * cfg.utd
            while update_tokens >= 1.0:
                ok = safe_update()
                if ok:
                    grad_steps += 1
                else:
                    out["diverged"] = True
                update_tokens -= 1.0
        wall = time.time() - t_start
        torch.cuda.synchronize()

        out["steady_wall_s"] = round(wall, 3)
        out["env_steps"] = env_steps
        out["grad_steps"] = grad_steps
        out["env_steps_per_sec"] = round(env_steps / wall, 1)
        out["grad_steps_per_sec"] = round(grad_steps / wall, 2)
        out["peak_gpu_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        out["reserved_gpu_mem_mb"] = round(torch.cuda.max_memory_reserved() / 1e6, 1)

        # CPU cores-equivalent + RSS for the whole process tree
        if proc is not None:
            try:
                cpu = proc.cpu_percent(None)
                rss = proc.memory_info().rss
                for ch in proc.children(recursive=True):
                    try:
                        cpu += ch.cpu_percent(None)
                        rss += ch.memory_info().rss
                    except Exception:
                        pass
                out["cpu_cores_equiv"] = round(cpu / 100.0, 2)
                out["rss_mb"] = round(rss / 1e6, 1)
            except Exception:
                pass

        out["ok"] = True
        try:
            env.close()
        except Exception:
            pass
    except RuntimeError as ex:
        msg = str(ex)
        out["error"] = f"{type(ex).__name__}: {msg[:300]}"
        out["oom"] = ("out of memory" in msg.lower()) or ("CUDA out of memory" in msg)
    except Exception as ex:
        out["error"] = f"{type(ex).__name__}: {str(ex)[:300]}"

    with open(args.worker_out, "w") as f:
        json.dump(out, f)
    return 0


# ===========================================================================
# DRIVER
# ===========================================================================
def _sample_system(stop_evt, gpu_indices, samples):
    """Background sampler: aggregate GPU util/mem and system CPU while a level runs."""
    import newt_bench_common as C
    try:
        import psutil
    except Exception:
        psutil = None
    if psutil is not None:
        psutil.cpu_percent(None)  # prime
    while not stop_evt.is_set():
        rec = {"t": time.time()}
        gutil, gmem = [], []
        for gi in gpu_indices:
            u = C.gpu_util_pct(gi)
            m = C.gpu_mem_used_mb(gi)
            if u is not None:
                gutil.append(u)
            if m is not None:
                gmem.append(m)
        rec["gpu_util"] = gutil
        rec["gpu_mem_used_mb"] = gmem
        if psutil is not None:
            rec["cpu_pct"] = psutil.cpu_percent(None)  # whole-system %
        samples.append(rec)
        stop_evt.wait(0.5)


def launch_level(config, h, l, num_envs, level, duration, warmup, compile_flag,
                 buffer_capacity, gpu_indices, tmpdir):
    """Launch `level` worker subprocesses concurrently; return aggregated metrics."""
    import newt_bench_common as C
    procs = []
    worker_outs = []
    for i in range(level):
        wout = os.path.join(tmpdir, f"worker_{config}_{h}h{l}l_L{level}_{i}.json")
        if os.path.exists(wout):
            os.remove(wout)
        worker_outs.append(wout)
        gpu = gpu_indices[i % len(gpu_indices)] if gpu_indices else 0
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("MUJOCO_GL", "egl")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker",
               "--config", config, "--h", str(h), "--l", str(l),
               "--num-envs", str(num_envs), "--duration", str(duration),
               "--warmup", str(warmup), "--buffer-capacity", str(buffer_capacity),
               "--worker-out", wout]
        if compile_flag:
            cmd.append("--compile")
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)

    # monitor while they run
    stop_evt = threading.Event()
    samples = []
    mon = threading.Thread(target=_sample_system, args=(stop_evt, gpu_indices or [0], samples))
    mon.start()

    t0 = time.time()
    timeout = 180 + duration * 4  # generous per-level ceiling
    for p in procs:
        remaining = max(1, timeout - (time.time() - t0))
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            p.kill()
    stop_evt.set()
    mon.join()

    # collect
    results = []
    for wout in worker_outs:
        if os.path.exists(wout):
            try:
                with open(wout) as f:
                    results.append(json.load(f))
            except Exception:
                results.append({"ok": False, "error": "unreadable worker output"})
        else:
            results.append({"ok": False, "error": "no worker output (killed/crashed)"})

    # aggregate
    oks = [r for r in results if r.get("ok")]
    any_oom = any(r.get("oom") for r in results)
    any_div = any(r.get("diverged") for r in results)
    agg = {
        "level": level,
        "num_workers_ok": len(oks),
        "oom": any_oom,
        "diverged": any_div,
        "agg_env_steps_per_sec": round(sum(r.get("env_steps_per_sec", 0) for r in oks), 1),
        "agg_grad_steps_per_sec": round(sum(r.get("grad_steps_per_sec", 0) for r in oks), 2),
        "mean_env_steps_per_sec": round(sum(r.get("env_steps_per_sec", 0) for r in oks) / max(1, len(oks)), 1),
        "mean_grad_steps_per_sec": round(sum(r.get("grad_steps_per_sec", 0) for r in oks) / max(1, len(oks)), 2),
        "mean_peak_gpu_mem_mb": round(sum(r.get("peak_gpu_mem_mb", 0) for r in oks) / max(1, len(oks)), 1) if oks else None,
        "sum_peak_gpu_mem_mb": round(sum(r.get("peak_gpu_mem_mb", 0) for r in oks), 1) if oks else None,
        "mean_env_build_s": round(sum(r.get("env_build_s", 0) for r in oks) / max(1, len(oks)), 2) if oks else None,
        "mean_first_reset_s": round(sum(r.get("first_reset_s", 0) for r in oks) / max(1, len(oks)), 2) if oks else None,
        "sum_cpu_cores_equiv": round(sum(r.get("cpu_cores_equiv", 0) for r in oks), 1) if oks else None,
        "sum_rss_gb": round(sum(r.get("rss_mb", 0) for r in oks) / 1e3, 2) if oks else None,
        "errors": [r.get("error") for r in results if r.get("error")],
    }
    # monitor aggregates
    if samples:
        def _flat(key):
            vals = []
            for s in samples:
                v = s.get(key)
                if isinstance(v, list):
                    vals.extend(v)
                elif v is not None:
                    vals.append(v)
            return vals
        gu = _flat("gpu_util")
        gm = _flat("gpu_mem_used_mb")
        cp = [s["cpu_pct"] for s in samples if "cpu_pct" in s]
        agg["gpu_util_mean_pct"] = round(sum(gu) / len(gu), 1) if gu else None
        agg["gpu_util_max_pct"] = round(max(gu), 1) if gu else None
        agg["gpu_mem_used_max_mb"] = round(max(gm), 1) if gm else None
        agg["system_cpu_mean_pct"] = round(sum(cp) / len(cp), 1) if cp else None
        agg["system_cpu_max_pct"] = round(max(cp), 1) if cp else None
    agg["_workers"] = results
    return agg


def classify_bottleneck(levels, hw, total_cores):
    """Given the per-level aggregates for one config, attribute the bottleneck."""
    ok_levels = [lv for lv in levels if lv["num_workers_ok"] == lv["level"]]
    oom_level = next((lv["level"] for lv in levels if lv.get("oom")), None)

    # throughput-optimal concurrency = level with max aggregate grad-steps/sec
    thru = [(lv["level"], lv["agg_grad_steps_per_sec"]) for lv in ok_levels if lv["agg_grad_steps_per_sec"] is not None]
    thr_opt = max(thru, key=lambda x: x[1])[0] if thru else (ok_levels[-1]["level"] if ok_levels else 1)

    # scaling efficiency: did going to the highest ok level keep improving aggregate?
    reason = []
    bottleneck = "unknown"
    if oom_level is not None:
        bottleneck = "gpu_memory"
        reason.append(f"CUDA OOM at concurrency {oom_level}")
    elif len(ok_levels) >= 2:
        lo, hi = ok_levels[0], ok_levels[-1]
        # aggregate throughput gain from lo->hi vs the level multiplier
        gain = (hi["agg_grad_steps_per_sec"] + 1e-9) / (lo["agg_grad_steps_per_sec"] + 1e-9)
        mult = hi["level"] / lo["level"]
        eff = gain / mult
        gpu_util = hi.get("gpu_util_mean_pct") or 0
        cpu_saturated = (hi.get("system_cpu_mean_pct") or 0) >= 85
        reason.append(f"agg grad/s {lo['agg_grad_steps_per_sec']}->{hi['agg_grad_steps_per_sec']} "
                      f"({hi['level']}x runs, scaling efficiency {eff:.2f})")
        reason.append(f"GPU util ~{gpu_util}%, system CPU ~{hi.get('system_cpu_mean_pct')}%")
        if eff < 0.6 and gpu_util < 50 and cpu_saturated:
            bottleneck = "env_cpu"
            reason.append("throughput stopped scaling while GPU idle and CPU pegged -> ENV/CPU-BOUND")
        elif eff < 0.6 and gpu_util >= 50:
            bottleneck = "gpu_compute"
            reason.append("throughput stopped scaling while GPU busy -> GPU-COMPUTE-BOUND")
        elif eff >= 0.8:
            bottleneck = "not_saturated"
            reason.append("throughput still scaling near-linearly -> not yet saturated at tested levels")
        else:
            bottleneck = "env_cpu" if cpu_saturated else "gpu_compute"
    return {
        "attributed_bottleneck": bottleneck,
        "throughput_optimal_concurrency": thr_opt,
        "memory_max_concurrency": (oom_level - 1) if oom_level else (ok_levels[-1]["level"] if ok_levels else 0),
        "reasoning": reason,
    }


def per_run_memory_ceiling(level1, hw):
    """Derived GPU-memory ceiling on runs-per-GPU from the level-1 peak."""
    if not hw["gpus"]:
        return None
    per_run = level1.get("mean_peak_gpu_mem_mb")
    if not per_run:
        return None
    gpu_total_mb = hw["gpus"][0]["total_mem_gb"] * 1000
    # leave ~10% headroom + a fixed CUDA context (~500MB) per process
    usable = gpu_total_mb * 0.9
    per_proc = per_run + 500
    return max(1, int(usable // per_proc))


def run_driver(args):
    import newt_bench_common as C
    hw = C.detect_hardware()
    total_cores = hw.get("cpu_count") or os.cpu_count()
    gpu_indices = list(range(hw["num_gpus"])) if hw["num_gpus"] > 0 else [0]

    print("=" * 78)
    print("Newt CONCURRENCY BENCHMARK (Script A)")
    print("=" * 78)
    print(json.dumps({k: hw[k] for k in ["gpus", "num_gpus", "mig_enabled", "cpu_count",
                                         "ram_total_gb", "torch_version", "cuda_version"]}, indent=2))
    print(f"\nNote: this machine has {hw['num_gpus']} GPU(s) and {total_cores} CPU cores. "
          f"Each run launches {args.num_envs} async env workers, so {args.num_envs} x concurrency "
          f"env processes compete for {total_cores} cores.")

    configs = [c.strip() for c in args.config.split(",") if c.strip()]
    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    levels = [lv for lv in levels if lv <= args.max_concurrency]
    cycles = {}
    for c in configs:
        cycles[c] = args.small_cycles if c == "small" else args.large_cycles

    tmpdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_worker_out")
    os.makedirs(tmpdir, exist_ok=True)

    report = {"hardware": hw, "params": vars(args), "configs": {}}

    for config in configs:
        h, l = cycles[config]
        print("\n" + "#" * 78)
        print(f"# CONFIG {config.upper()}  ({h}h{l}l, num_envs={args.num_envs})")
        print("#" * 78)
        level_aggs = []
        for level in levels:
            # safety: don't oversubscribe the box into the ground
            if level * args.num_envs > args.max_env_processes:
                print(f"  [skip] concurrency {level} would spawn {level*args.num_envs} env procs "
                      f"(> --max-env-processes {args.max_env_processes})")
                break
            print(f"\n  -> concurrency {level} ({level} run(s), "
                  f"{level*args.num_envs} env workers) ...", flush=True)
            agg = launch_level(config, h, l, args.num_envs, level, args.duration,
                               args.warmup, args.compile, args.buffer_capacity,
                               gpu_indices, tmpdir)
            level_aggs.append(agg)
            print(f"     ok={agg['num_workers_ok']}/{level}  "
                  f"agg_env/s={agg['agg_env_steps_per_sec']}  agg_grad/s={agg['agg_grad_steps_per_sec']}  "
                  f"mean_peak_mem={agg['mean_peak_gpu_mem_mb']}MB  "
                  f"GPUutil~{agg.get('gpu_util_mean_pct')}%  CPU~{agg.get('system_cpu_mean_pct')}%  "
                  f"CPUcores~{agg.get('sum_cpu_cores_equiv')}")
            if agg["errors"]:
                print(f"     errors: {agg['errors'][:2]}")
            if agg.get("oom"):
                print(f"     >> OOM at concurrency {level}; stopping escalation for {config}.")
                break
            # stop if throughput clearly saturated (no >5% gain over previous level)
            if len(level_aggs) >= 2:
                prev = level_aggs[-2]["agg_grad_steps_per_sec"]
                cur = agg["agg_grad_steps_per_sec"]
                if prev and cur <= prev * 1.05:
                    print(f"     >> aggregate throughput saturated ({prev} -> {cur} grad/s); stopping.")
                    break

        clsf = classify_bottleneck(level_aggs, hw, total_cores)
        ceiling = per_run_memory_ceiling(level_aggs[0], hw) if level_aggs else None
        clsf["gpu_memory_ceiling_runs_per_gpu"] = ceiling
        report["configs"][config] = {"cycles": f"{h}h{l}l", "levels": level_aggs, "analysis": clsf}

        # per-config summary
        print(f"\n  --- {config.upper()} summary ---")
        print(f"    attributed bottleneck      : {clsf['attributed_bottleneck']}")
        print(f"    throughput-optimal runs    : {clsf['throughput_optimal_concurrency']} per GPU")
        print(f"    GPU-memory ceiling (derived): {ceiling} runs/GPU "
              f"(per-run peak ~{level_aggs[0].get('mean_peak_gpu_mem_mb') if level_aggs else '?'}MB)")
        for r in clsf["reasoning"]:
            print(f"      - {r}")

    # ---- machine-readable output + scheduler recommendation ---------------
    sched = {}
    for config, cd in report["configs"].items():
        a = cd["analysis"]
        # recommend the throughput-optimal, but never above the memory ceiling
        rec = a["throughput_optimal_concurrency"]
        if a.get("gpu_memory_ceiling_runs_per_gpu"):
            rec = min(rec, a["gpu_memory_ceiling_runs_per_gpu"])
        sched[config] = {
            "runs_per_gpu": rec,
            "bottleneck": a["attributed_bottleneck"],
            "cycles": cd["cycles"],
        }
    report["scheduler_recommendation"] = sched

    out_path = args.out
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 78)
    print("RECOMMENDATION (runs per GPU for a downstream scheduler)")
    print("=" * 78)
    for config, s in sched.items():
        print(f"  {config.upper():6s}: {s['runs_per_gpu']} run(s)/GPU   "
              f"[bottleneck: {s['bottleneck']}]")
    hint = ("Because each run's env workers dominate, reducing --num-envs per run "
            "(fewer async workers) is the main lever for packing more runs per GPU "
            "when the bottleneck is env/cpu; it does little when the bottleneck is gpu_memory.")
    print("\n  " + hint)
    print(f"\nWrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Newt concurrency benchmark (Script A).")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    # worker args
    ap.add_argument("--config", default="small,large")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--l", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=21)
    ap.add_argument("--duration", type=float, default=8.0, help="steady-state seconds per worker")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--buffer-capacity", type=int, default=200_000)
    ap.add_argument("--compile", action="store_true", help="use torch.compile (matches real runs; slower cold start)")
    ap.add_argument("--worker-out", default=None)
    # driver args
    ap.add_argument("--levels", default="1,2,4,8")
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--max-env-processes", type=int, default=200,
                    help="safety cap: skip a level if concurrency*num_envs exceeds this")
    ap.add_argument("--small-cycles", default="1h1l", help="HxL for SMALL (stable default so grad steps run)")
    ap.add_argument("--large-cycles", default="1h1l", help="HxL for LARGE")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json"))
    args = ap.parse_args()

    # parse cycle strings for driver
    def _pc(s):
        s = s.lower().split("h")
        return int(s[0]), int(s[1].rstrip("l"))
    args.small_cycles = _pc(args.small_cycles)
    args.large_cycles = _pc(args.large_cycles)

    if args.worker:
        if args.config not in ("small", "large"):
            print("worker needs a single --config small|large", file=sys.stderr)
            return 2
        return run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
