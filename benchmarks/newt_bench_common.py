"""
Shared repo-inspection / setup helpers for the two Newt benchmark scripts:

  - bench_concurrency.py  (Script A: how many runs fit on the GPU(s))
  - profile_components.py  (Script B: where per-run wall-clock time goes)

Both scripts import this module, but each is runnable on its own. Nothing here
launches a benchmark by itself; it only knows how to faithfully construct the
*real* Newt objects (config, vectorized env, WorldModel + TDMPC2 agent, replay
buffer) exactly the way tdmpc2/train.py does, and how to seed the buffer with
real env rollouts so agent.update() runs on a realistic latent distribution.

Why we drive the real objects instead of `python train.py`:
  * train.py spends ~seeding_coef * num_envs * episode_length (=52,500 for the
    default dmcontrol run) pure env steps before the FIRST gradient step, plus a
    step-0 eval. That is unusable for a short, controlled probe.
  * There is no --max-steps / --benchmark flag in the entry point.
  * We still exercise the identical code path: envs.make_env(), WorldModel,
    TDMPC2.update()/.plan(), common.buffer.Buffer -- just wired up directly so we
    control step budget, teardown, and instrumentation.

Grounded in the repo's own experiment scripts (experiments/paper/maskx/*.sh,
experiments/local_3seed.sh), the representative run is:
    task=dmcontrol num_envs=21 model_size=S use_trm_dynamics=simple obs=state
and those scripts launch 3 seeds concurrently on one GPU (`... & ... & wait`),
which is exactly the question Script A answers.
"""

import os
import sys
import time
import json
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

# train.py sets these before importing torch; mirror the important ones so the
# subprocess/env behaviour matches a real run.
os.environ.setdefault("LAZY_LEGACY_OP", "0")


def setup_mujoco_gl(force=None, verbose=True):
    """Select a working MuJoCo GL backend and export it via MUJOCO_GL.

    dm_control/MuJoCo needs a GL backend to initialise even for state-only runs.
    On headless nodes without EGL (e.g. the 42 cluster, whose own scripts export
    MUJOCO_GL=disable) trying to use EGL crashes env construction. So:

      1. If the caller/cluster already set MUJOCO_GL, respect it (no probe).
      2. Otherwise attempt EGL in an isolated subprocess (a failed GL init can
         segfault rather than raise, so we don't probe in-process); if the probe
         fails, fall back to MUJOCO_GL=disable (state-only, headless).

    Runs once at import; worker subprocesses inherit the resolved MUJOCO_GL through
    os.environ, so they never re-probe. Override with force="egl"/"disable"/... ."""
    pre_set = os.environ.get("MUJOCO_GL")
    if force:
        backend = force
    elif pre_set:
        backend = pre_set
    else:
        backend = "egl" if _egl_probe_ok() else "disable"
        if verbose and backend == "disable":
            print("[newt_bench] EGL probe failed -> MUJOCO_GL=disable (state-only headless).")
    os.environ["MUJOCO_GL"] = backend
    if backend == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    else:
        os.environ.pop("PYOPENGL_PLATFORM", None)
    return backend


def _egl_probe_ok(timeout=60):
    """True iff dm_control can actually initialise a headless EGL display here.

    We must probe dm_control's OWN EGL renderer (PyOpenGL EGL), not mujoco's
    GLContext: on some nodes (e.g. the 42 cluster A100s) mujoco.GLContext succeeds
    yet dm_control raises 'Cannot initialize a headless EGL display' at env build.
    So the probe loads a dm_control suite env with MUJOCO_GL=egl and forces a
    render, which drives the exact code path the real envs use.

    Isolated in a subprocess so a hard GL failure can't take down the caller; we
    os._exit(0) after success to skip the (occasionally throwing) EGL teardown and
    key off a stdout sentinel rather than the exit code."""
    code = (
        "import os, sys\n"
        "os.environ['MUJOCO_GL'] = 'egl'\n"
        "from dm_control import suite\n"
        "e = suite.load('cartpole', 'balance'); e.reset()\n"
        "e.physics.render(64, 64, camera_id=0)\n"
        "print('EGL_OK'); sys.stdout.flush(); os._exit(0)\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=timeout)
        return "EGL_OK" in r.stdout
    except Exception:
        return False


# Resolve the GL backend at import so every entry point (driver, worker, profiler)
# and any later dm_control import sees a working MUJOCO_GL.
setup_mujoco_gl()

# Make `tdmpc2/` importable regardless of where the script is invoked from.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDMPC2_DIR = os.path.join(REPO_ROOT, "tdmpc2")
if TDMPC2_DIR not in sys.path:
    sys.path.insert(0, TDMPC2_DIR)


# --------------------------------------------------------------------------- #
# Hardware / system inspection
# --------------------------------------------------------------------------- #
def _run(cmd: List[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return ""


def detect_hardware() -> Dict[str, Any]:
    """GPU count / VRAM / MIG status, CPU cores, system RAM.

    Reported by both scripts so the numbers are self-describing on whatever
    machine they run on (no single-card / A100 assumption is baked in)."""
    import torch

    info: Dict[str, Any] = {}
    info["cpu_count"] = os.cpu_count()
    try:
        import multiprocessing
        info["cpu_count"] = multiprocessing.cpu_count()
    except Exception:
        pass

    # System RAM (GB)
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        info["ram_available_gb"] = round(psutil.virtual_memory().available / 1e9, 1)
    except Exception:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            psize = os.sysconf("SC_PAGE_SIZE")
            info["ram_total_gb"] = round(pages * psize / 1e9, 1)
        except Exception:
            info["ram_total_gb"] = None

    info["cuda_available"] = torch.cuda.is_available()
    info["torch_version"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_mem_gb": round(props.total_memory / 1e9, 2),
                "sm_count": props.multi_processor_count,
                "capability": f"{props.major}.{props.minor}",
            })
    info["gpus"] = gpus
    info["num_gpus"] = len(gpus)

    # MIG detection via nvidia-smi -L (MIG devices show "MIG ... Device" lines).
    smi_l = _run(["nvidia-smi", "-L"])
    info["mig_enabled"] = ("MIG" in smi_l and "Device" in smi_l)
    info["nvidia_smi_L"] = smi_l.strip()
    return info


def gpu_mem_used_mb(index: int = 0) -> Optional[float]:
    """Total memory *used* on a physical GPU (all processes), via nvidia-smi.
    Used by Script A to see aggregate pressure across concurrent runs."""
    out = _run([
        "nvidia-smi", f"--query-gpu=memory.used",
        "--format=csv,noheader,nounits", "-i", str(index),
    ])
    try:
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


def gpu_util_pct(index: int = 0) -> Optional[float]:
    out = _run([
        "nvidia-smi", "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits", "-i", str(index),
    ])
    try:
        return float(out.strip().splitlines()[0])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Config construction (faithful to tdmpc2/train.py + config.parse_cfg)
# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    """A named model/dynamics configuration to benchmark."""
    name: str
    latent_dim: int
    H_cycles: int
    L_cycles: int
    use_film_dynamics: bool
    xl_dynamics_mlp: bool
    use_trm_dynamics: Optional[str] = "simple"
    model_size: str = "S"
    num_envs: int = 21
    task: str = "dmcontrol"
    extra: Dict[str, Any] = field(default_factory=dict)


# The two representative configs the prompt asks for. Grounded on the repo's
# maskx / local_3seed experiment scripts (model_size=S, simple dynamics), with
# latent_dim + XL-dynamics being the parameter-count knob that drives VRAM.
def small_spec(h=2, l=6, num_envs=21) -> RunSpec:
    # latent_dim=16, no XL dynamics -> the "smp_..._16ld_..._maskx_film" family.
    return RunSpec(name="SMALL", latent_dim=16, H_cycles=h, L_cycles=l,
                   use_film_dynamics=True, xl_dynamics_mlp=False, num_envs=num_envs)


def large_spec(h=8, l=4, num_envs=21) -> RunSpec:
    # latent_dim=512 + XL dynamics MLP. xl_dynamics_mlp only takes effect on the
    # NON-FiLM SimpleTRM path (a [512,512] hidden MLP, common/trm/simple_trm.py),
    # so LARGE turns FiLM off to actually exercise the big dynamics net.
    return RunSpec(name="LARGE", latent_dim=512, H_cycles=h, L_cycles=l,
                   use_film_dynamics=False, xl_dynamics_mlp=True, num_envs=num_envs)


def build_cfg(spec: RunSpec, compile: bool = True, seed: int = 0,
              log_trm_gradnorms: bool = False):
    """Build a fully-parsed Config object exactly as train.py would, then apply
    the spec's overrides *after* parse_cfg.

    The post-parse override is essential: with use_trm_dynamics='simple' and
    model_size='S', config.parse_cfg() applies TRM_SIZE['S'], which forces
    latent_dim=384 / H_cycles=4 / L_cycles=3. The real experiment scripts defeat
    that by passing the values on the Hydra CLI (which parse_cfg treats as
    cli_overrides). We have no Hydra CLI here, so we overwrite the handful of
    fields we care about on the returned object instead -- same end state."""
    from omegaconf import OmegaConf
    from config import Config, parse_cfg
    import hydra.utils as hu

    # parse_cfg calls hydra.utils.get_original_cwd(); no Hydra app is running, so
    # point it at the repo root (that is only used to build a logs/ work_dir).
    hu.get_original_cwd = lambda: REPO_ROOT

    cfg = OmegaConf.structured(Config)
    cfg.task = spec.task
    cfg.obs = "state"
    cfg.num_envs = spec.num_envs
    cfg.env_mode = "async"
    cfg.model_size = spec.model_size
    cfg.use_trm_encoder = False
    cfg.use_trm_dynamics = spec.use_trm_dynamics
    cfg.use_task_embedding = True
    cfg.wm_regularization_type = "sigreg"
    cfg.use_simple_trm_skip_connections = True
    cfg.simple_trm_skip_type = "swiglu"
    cfg.rrm_mask_x_for_y_update = True
    cfg.use_dis_loss = True
    cfg.enable_wandb = False
    cfg.save_agent = False
    cfg.save_video = False
    cfg.multiproc = False
    cfg.compile = compile
    cfg.log_trm_gradnorms = log_trm_gradnorms
    cfg.seed = seed

    cfg = parse_cfg(cfg)  # returns a plain Config (OmegaConf.to_object)

    # ---- post-parse overrides (defeat TRM_SIZE preset clobbering) ----
    cfg.latent_dim = spec.latent_dim
    cfg.H_cycles = spec.H_cycles
    cfg.L_cycles = spec.L_cycles
    cfg.use_film_dynamics = spec.use_film_dynamics
    cfg.xl_dynamics_mlp = spec.xl_dynamics_mlp
    cfg.srm_truncation_length = min(cfg.srm_truncation_length, spec.L_cycles)
    for k, v in spec.extra.items():
        setattr(cfg, k, v)

    cfg.rank = 0
    cfg.world_size = 1
    cfg.batch_size = 128
    return cfg


def build_env(cfg):
    """Build the real vectorized multitask env; fills obs_shape/action_dim/
    episode_length on cfg (exactly like train.py's make_env call)."""
    from envs import make_env
    return make_env(cfg)


def build_agent(cfg):
    """Build the real WorldModel + TDMPC2 agent on cuda:<rank> (no DDP)."""
    import torch
    from common.world_model import WorldModel
    from tdmpc2 import TDMPC2
    torch.cuda.set_device(cfg.rank)
    model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
    agent = TDMPC2(model, cfg)
    return agent


def build_buffer(cfg, capacity: int = 200_000):
    from common.buffer import Buffer
    return Buffer(capacity=capacity, batch_size=cfg.batch_size,
                  horizon=cfg.horizon, multiproc=False, compile=False)


def seed_buffer_with_rollouts(env, cfg, buffer, num_episodes_worth: int = 1):
    """Collect `num_episodes_worth` full episodes per env with RANDOM actions and
    add them to the buffer. This mirrors trainer.py's seeding phase and gives
    agent.update() a realistic latent distribution (synthetic gaussian obs make
    the value-target two-hot bins overflow -> NaN -> device asserts).

    Returns (env_build/step timing dict). All dmcontrol episodes are exactly
    episode_length and never terminate early, so one pass = num_envs episodes."""
    import torch
    from tensordict import TensorDict

    N = cfg.num_envs
    EL = cfg.episode_length
    tasks = torch.arange(N, dtype=torch.int32)

    for _ in range(num_episodes_worth):
        obs, info = env.reset()
        obs_buf = [obs.clone()]
        act_buf = []
        rew_buf = []
        for _s in range(EL):
            a = env.rand_act()
            o, r, term, trunc, info = env.step(a)
            _o = o.clone()
            done = term | trunc
            if "final_observation" in info and done.any():
                _o[done] = info["final_observation"]
            obs_buf.append(_o)
            act_buf.append(a.clone())
            rew_buf.append(r.clone())
        O = torch.stack(obs_buf)                                             # [EL+1, N, 128]
        nan_a = torch.full_like(act_buf[0], float("nan"))
        nan_r = torch.full_like(rew_buf[0], float("nan"))
        A = torch.stack([nan_a] + act_buf)                                  # [EL+1, N, 16]
        R = torch.stack([nan_r] + rew_buf)                                  # [EL+1, N]
        for i in range(N):
            td = TensorDict(dict(
                obs=O[:, i], action=A[:, i], reward=R[:, i],
                task=tasks[i].repeat(EL + 1)), batch_size=(EL + 1,))
            buffer.add(td.unsqueeze(0))
    return buffer


# --------------------------------------------------------------------------- #
# CUDA-event timing helper (shared by Script B, handy for A too)
# --------------------------------------------------------------------------- #
class CudaTimer:
    """Accumulates GPU wall-time per named tag using CUDA events, so async kernel
    execution isn't misattributed. Overhead is a pair of event records per call
    (~microseconds) and can be disabled by constructing with enabled=False."""

    def __init__(self, enabled: bool = True):
        import torch
        self.torch = torch
        self.enabled = enabled and torch.cuda.is_available()
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._stack: List[Any] = []

    class _Scope:
        def __init__(self, parent, tag):
            self.parent = parent
            self.tag = tag

        def __enter__(self):
            if not self.parent.enabled:
                return self
            torch = self.parent.torch
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return self

        def __exit__(self, *exc):
            if not self.parent.enabled:
                return False
            self.end.record()
            self.end.synchronize()
            ms = self.start.elapsed_time(self.end)
            self.parent.totals[self.tag] = self.parent.totals.get(self.tag, 0.0) + ms
            self.parent.counts[self.tag] = self.parent.counts.get(self.tag, 0) + 1
            return False

    def scope(self, tag: str):
        return CudaTimer._Scope(self, tag)

    def reset(self):
        self.totals.clear()
        self.counts.clear()
