"""Recursion-trajectory analysis for the recursive dynamics models (SimpleTRM / TRM / SRM).

For a wandb run name or run group (multiple seeds of the same config), this script:
  1. pulls the most recent saved model artifact for each run (cached on disk),
  2. rebuilds the agent from the run's stored config and runs eval episodes,
  3. records the dynamics latent state at every recursion step (both the low-level
     carry, updated every inner L cycle, and the high-level carry, updated every
     outer H cycle),
  4. fits PCA over all tasks and renders, per seed, a multi-panel figure showing the
     reasoning trajectory in PC space alongside the per-step cosine similarity and
     state difference, plus a cross-seed summary figure and tidy CSVs.

Eval transitions are cached per checkpoint, so re-plotting does not re-run the envs.
Outputs land in <repo>/analysis/recursion_pca/<group>/ (with an `_ep<N>` suffix when
--episodes != 1, so runs at different eval depths never overwrite each other) and
caches in <repo>/analysis/cache/.

Usage (from anywhere, in the newt conda env):
    python analysis/analyze_recursion.py --group paper_simple_384ld_4h3l
    python analysis/analyze_recursion.py --run smp_s0_384ld_4h3l_simnorm --episodes 10
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--group', help='wandb group name (all runs/seeds in the group are analyzed)')
    src.add_argument('--run', help='wandb run display name (a single run)')
    p.add_argument('--entity', default='trm-dynamics')
    p.add_argument('--project', default='TRM Dynamics')
    p.add_argument('--episodes', type=int, default=1, help='eval episodes per parallel env')
    p.add_argument('--max-transitions', type=int, default=64,
                   help='transitions per task fed through the recursion recorder')
    p.add_argument('--seed', type=int, default=42, help='seed for eval rollouts and subsampling')
    p.add_argument('--compile', action='store_true',
                   help='enable torch.compile for the eval rollout (fast steady-state, slow warmup; '
                        'default is fully eager, which is faster for a few episodes)')
    p.add_argument('--out-dir', default=None, help='output directory (default: <repo>/analysis/recursion_pca/<group>)')
    p.add_argument('--tsne-points', type=int, default=4000,
                   help='max recursion states embedded by t-SNE per carry (subsampled; 0 disables t-SNE)')
    p.add_argument('--tsne-perplexity', type=float, default=30.0)
    return p.parse_args()


ARGS = parse_args()

# Environment setup must precede torch / env imports (mirrors train.py).
os.environ['MUJOCO_GL'] = os.getenv('MUJOCO_GL', 'disable')  # state-only eval, no rendering
os.environ['LAZY_LEGACY_OP'] = '0'
if not ARGS.compile:
    # Neutralize *all* torch.compile decorators (TRMBlock.forward, SRM.forward, ...);
    # must be set before torch is imported.
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
import warnings
warnings.filterwarnings('ignore')

import json
import re
import sys
from dataclasses import fields
from pathlib import Path

# The script lives in <repo>/analysis/ but imports the training code from <repo>/tdmpc2/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tdmpc2'))

import numpy as np
import torch
from tqdm import tqdm

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import matplotlib
matplotlib.use('Agg')
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from common import set_seed
from common.world_model import WorldModel
from config import Config
from envs import make_env
from tdmpc2 import TDMPC2

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / 'analysis' / 'cache'

# --- figure / font sizing & style (matches the fanda paper scripts) ------------
FIG_WIDTH_IN = 4      # width of ONE panel
FIG_HEIGHT_IN = 3
FONT_PT = 11
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['savefig.dpi'] = 300
# Roboto matches the thesis template's sans (see fanda/visualizations.py).
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Roboto', 'DejaVu Sans', 'sans-serif']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Roboto'
plt.rcParams['mathtext.it'] = 'Roboto:italic'
plt.rcParams['mathtext.bf'] = 'Roboto:bold'


ARCH_NAME = {'simple': 'SimpleTRM', 'srm': 'SRM', 'trm': 'TRM'}


def shade(color, frac, lo=0.2):
    """Blend `color` toward white by recursion progress: frac=0 (initial state) is
    lightest, frac=1 (final cycle) is the full color."""
    c = np.asarray(matplotlib.colors.to_rgb(color))
    t = lo + (1.0 - lo) * frac
    return tuple(1.0 + (c - 1.0) * t)


# ------------------------------------------------------------------ wandb pulls

def parse_run_config(run) -> dict:
    """run.config round-trips the full post-parse Config; older wandb clients return
    it as a raw JSON string with {"value": ...} wrappers."""
    cfg = run.config
    if isinstance(cfg, str):
        cfg = json.loads(cfg)
    return {k: (v['value'] if isinstance(v, dict) and set(v) <= {'value', 'desc'} else v)
            for k, v in cfg.items()}


def latest_model_artifact(run):
    arts = [a for a in run.logged_artifacts() if a.type == 'model']
    if not arts:
        return None
    return max(arts, key=lambda a: a.created_at)


def download_checkpoint(artifact) -> Path:
    """Download (or reuse a cached copy of) a model artifact; returns the .pt path."""
    dest = CACHE_DIR / 'checkpoints' / artifact.name.replace(':', '@').replace('/', '_')
    if not list(dest.glob('*.pt')):
        print(f'  downloading {artifact.name} ...')
        artifact.download(root=str(dest))
    pts = list(dest.glob('*.pt'))
    assert pts, f'No .pt file found in artifact {artifact.name}'
    return pts[0]


def checkpoint_step(artifact, cfg) -> int:
    """Training step at which the checkpoint was saved (identifier is '<step:,>' or
    'final' with ',' -> '_' from artifact-name sanitization); used for the planner's
    constrained-planning schedule."""
    ident = artifact.name.split(':')[0].split('-')[-1]
    digits = ident.replace('_', '')
    return int(digits) if digits.isdigit() else cfg.steps


# ------------------------------------------------------------------ agent setup

def build_cfg(run_config: dict) -> Config:
    """Rebuild the training Config from the stored (fully parsed) run config, with
    runtime fields overridden for local single-GPU eval."""
    names = {f.name for f in fields(Config)}
    cfg = Config(**{k: v for k, v in run_config.items() if k in names})
    cfg.rank = 0
    cfg.world_size = 1
    cfg.multiproc = False
    cfg.enable_wandb = False
    cfg.save_video = False
    cfg.checkpoint = None
    cfg.use_demos = False
    cfg.child_env = False
    if not ARGS.compile:
        cfg.compile = False
        cfg.compile_planning = False
    # wandb stores tuples as lists; normalize for the no-env (cached-transitions) path.
    if isinstance(cfg.obs_shape, dict):
        cfg.obs_shape = {k: tuple(v) for k, v in cfg.obs_shape.items()}
    return cfg


def build_agent(cfg: Config, ckpt: Path) -> TDMPC2:
    try:
        model = WorldModel(cfg).to('cuda:0')
        agent = TDMPC2(model, cfg)
        agent.load(str(ckpt))
    except RuntimeError as e:
        # Legacy checkpoints: runs from before the L_layers refactor stored
        # L_layers from the TRM presets (unused by the then-hardcoded
        # single-layer SimpleTRM/SRM cores). The current code sizes the core
        # from L_layers, so rebuild with the equivalent depth of 1.
        if cfg.L_layers == 1 or '_dynamics' not in str(e):
            raise
        print(f'  state_dict mismatch with L_layers={cfg.L_layers}; '
              f'retrying with L_layers=1 (legacy single-layer core)')
        del model
        torch.cuda.empty_cache()
        cfg.L_layers = 1
        model = WorldModel(cfg).to('cuda:0')
        agent = TDMPC2(model, cfg)
        agent.load(str(ckpt))
    agent.model.eval()
    return agent


# ------------------------------------------------------------------ eval rollout

def task_vector(cfg) -> torch.Tensor:
    """Env-index -> global-task-id mapping, exactly as Trainer sets it up (rank 0, ws 1)."""
    tasks = torch.tensor(list(range(cfg.num_global_tasks)), dtype=torch.int32)
    if cfg.task != 'soup' and len(tasks) < cfg.num_envs:
        tasks = tasks.repeat_interleave(cfg.num_envs // len(tasks))
    return tasks


@torch.no_grad()
def collect_transitions(cfg, agent, tasks, step, episodes):
    """Run eval episodes and return executed (obs, action, task) transitions plus
    per-task episode rewards. Mirrors Trainer.eval()."""
    env = make_env(cfg)
    pbar = tqdm(total=episodes * cfg.num_envs, desc='  eval episodes', unit='ep')
    try:
        obs, _ = env.reset()
        n = cfg.num_envs
        ep_len = torch.zeros(n)
        ep_reward = torch.zeros(n)
        done_count = torch.zeros(n, dtype=torch.int32)
        obs_list, act_list, active_list = [], [], []
        rewards = {t: [] for t in range(cfg.num_global_tasks)}

        while (done_count < episodes).any():
            torch.compiler.cudagraph_mark_step_begin()
            action = agent(obs, t0=ep_len == 0, step=step, eval_mode=True, task=tasks, mpc=True)
            obs_list.append(obs.clone())
            act_list.append(action.clone())
            active_list.append(done_count < episodes)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            ep_reward += reward
            ep_len += 1
            for i in torch.nonzero(done).flatten().tolist():
                rewards[tasks[i].item()].append(ep_reward[i].item())
                ep_reward[i] = 0.0
                ep_len[i] = 0.0
                done_count[i] += 1
            # A fast env can finish more than `episodes` episodes while stragglers
            # catch up; count each env's progress only up to its quota.
            pbar.update(int(done_count.clamp(max=episodes).sum()) - pbar.n)
    finally:
        pbar.close()
        env.close()

    obs_all = torch.stack(obs_list)        # [S, n, obs_dim]
    act_all = torch.stack(act_list)        # [S, n, action_dim]
    active = torch.stack(active_list)      # [S, n]
    task_all = tasks.unsqueeze(0).expand(active.shape)
    keep = active.flatten()
    return (obs_all.flatten(0, 1)[keep].numpy(),
            act_all.flatten(0, 1)[keep].numpy(),
            task_all.flatten(0, 1)[keep].numpy().astype(np.int32),
            {t: float(np.mean(r)) for t, r in rewards.items() if r})


def get_transitions(cfg, agent, tasks, step, artifact_name):
    """Cached wrapper around collect_transitions, keyed on checkpoint + eval params."""
    key = re.sub(r'[^A-Za-z0-9_.@-]', '_', f'{artifact_name}_ep{ARGS.episodes}_seed{ARGS.seed}')
    fp = CACHE_DIR / 'transitions' / f'{key}.npz'
    if fp.exists():
        print(f'  using cached transitions {fp.name}')
        d = np.load(fp)
        rewards = json.loads(str(d['rewards_json']))
        return d['obs'], d['act'], d['task'], {int(k): v for k, v in rewards.items()}
    print(f'  running eval rollouts ({ARGS.episodes} episode(s) x {cfg.num_envs} envs) ...')
    obs, act, task, rewards = collect_transitions(cfg, agent, tasks, step, ARGS.episodes)
    fp.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(fp, obs=obs, act=act, task=task,
                        rewards_json=json.dumps({str(k): v for k, v in rewards.items()}))
    return obs, act, task, rewards


# ------------------------------------------------------- recursion-step recording
# Each recorder replays the dynamics model's forward pass eagerly under no_grad
# (the train-time detaches are no-ops here) and snapshots the latent after every
# update. Returns (z_seq [N, H*L+1, D], y_seq [N, H+1, D]) including the initial state.

@torch.no_grad()
def record_simple(model, z, a, task):
    m = model._dynamics
    x = torch.cat([model.task_emb(z, task), a], dim=-1)
    carry = m.initial_carry(x)
    H, L = m.config.H_cycles, m.config.L_cycles
    film_params = m._film_params(carry.x) if m.use_film else None
    z_seq, y_seq = [carry.z.clone()], [carry.y.clone()]
    for _ in range(H):
        for _ in range(L):
            z_before = carry.z
            carry.z = m._apply_fn(carry, film_params)
            if m.use_skip:
                carry.z = m._skip(carry.z, z_before)
            z_seq.append(carry.z.clone())
        y_before = carry.y
        carry.y = m._apply_fn(carry, film_params, mask_x=m.mask_x_for_y)
        if m.use_skip:
            carry.y = m._skip(carry.y, y_before)
        y_seq.append(carry.y.clone())
    return torch.stack(z_seq, 1), torch.stack(y_seq, 1)


@torch.no_grad()
def record_srm(model, z, a, task):
    m = model._dynamics
    x = torch.cat([model.task_emb(z, task), a], dim=-1)
    carry = m.initial_carry(x)
    film_params = m._film_params(carry.x) if m.use_film else None
    z_seq, y_seq = [carry.z.clone()], [carry.z.clone()]
    for _ in range(m.H_cycles):
        carry.context = m._compute_context(carry.z)
        for _ in range(m.L_cycles):
            carry = m._apply_fun(carry, film_params)
            z_seq.append(carry.z.clone())
        y_seq.append(carry.z.clone())
    return torch.stack(z_seq, 1), torch.stack(y_seq, 1)


@torch.no_grad()
def record_trm(model, z, a, task):
    inner = model._dynamics.inner
    obs = torch.cat([z, a], dim=-1)
    task_flat = model.reshape_task_ids(task, obs.shape[:-1])
    B, cfg = obs.shape[0], inner.config
    cos_sin = inner._get_cos_sin()
    inject_base = inner._input_embeddings(obs, task_flat)
    z_H = inner.H_init.expand(B, cfg.seq_len, -1).clone()
    z_L = inner.L_init.expand(B, cfg.seq_len, -1).clone()
    # The token-level carries live in hidden space; snapshot them through the output
    # head (pool -> lm_head -> latent norm) so every recorded state is a WM latent.
    decode = lambda t: inner.lm_head_norm(inner.lm_head(inner._pool(t)).to(torch.float32))
    z_seq, y_seq = [decode(z_L)], [decode(z_H)]
    for _ in range(cfg.H_cycles):
        z_H_inject = z_H + inject_base
        for _ in range(cfg.L_cycles):
            z_L = inner.L_level(z_L, z_H_inject, cos_sin=cos_sin)
            z_seq.append(decode(z_L))
        z_H = inner.L_level(z_H, z_L, cos_sin=cos_sin)
        y_seq.append(decode(z_H))
    return torch.stack(z_seq, 1), torch.stack(y_seq, 1)


RECORDERS = {'simple': record_simple, 'srm': record_srm, 'trm': record_trm}


def record_recursion(cfg, agent, obs, act, task, rng):
    """Subsample transitions per task, encode, and replay the recursion. Returns
    (z_seq, y_seq, task_ids) as numpy arrays."""
    arch = cfg.use_trm_dynamics
    assert arch in RECORDERS, \
        f'use_trm_dynamics={arch!r} has no recursion to analyze (plain MLP dynamics).'
    idx = []
    for t in np.unique(task):
        t_idx = np.nonzero(task == t)[0]
        take = min(len(t_idx), ARGS.max_transitions)
        idx.append(rng.choice(t_idx, size=take, replace=False))
    idx = np.concatenate(idx)

    device = agent.device
    model = agent.model
    z_chunks, y_chunks = [], []
    for chunk in tqdm(np.array_split(idx, max(1, len(idx) // 512)),
                      desc='  recording recursion', unit='chunk'):
        obs_t = torch.as_tensor(obs[chunk], dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act[chunk], dtype=torch.float32, device=device)
        task_t = torch.as_tensor(task[chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            z0 = model.encode(obs_t, task_t)
            z_seq, y_seq = RECORDERS[arch](model, z0, act_t, task_t)
        z_chunks.append(z_seq.float().cpu().numpy())
        y_chunks.append(y_seq.float().cpu().numpy())
    return np.concatenate(z_chunks), np.concatenate(y_chunks), task[idx]


# ------------------------------------------------------------------ analysis

def pca_fit(X, k=2):
    """PCA via SVD; returns (mean, components [k, D], explained variance ratio [k])."""
    X = X.astype(np.float64)
    mu = X.mean(0)
    _, s, vt = np.linalg.svd(X - mu, full_matrices=False)
    evr = s**2 / np.sum(s**2)
    return mu, vt[:k], evr[:k]


def step_metrics(seq):
    """Per-transition cosine similarity and L2 difference between successive states.
    seq: [N, T, D] -> (cossim [N, T-1], delta [N, T-1])."""
    a, b = seq[:, 1:], seq[:, :-1]
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den, np.linalg.norm(a - b, axis=-1)


def tsne_embed(seq, task_ids, seed):
    """t-SNE over a subsample of all recursion states (all tasks x cycles pooled).
    seq: [N, T, D] -> (emb [M, 2], task of each point [M], cycle of each point [M])."""
    from sklearn.manifold import TSNE
    N, T, D = seq.shape
    rng = np.random.default_rng(seed)
    idx = np.arange(N * T)
    if ARGS.tsne_points and len(idx) > ARGS.tsne_points:
        idx = rng.choice(idx, size=ARGS.tsne_points, replace=False)
    X = seq.reshape(-1, D)[idx].astype(np.float64)
    perplexity = min(ARGS.tsne_perplexity, (len(idx) - 1) / 3)
    emb = TSNE(n_components=2, perplexity=perplexity, init='pca',
               learning_rate='auto', random_state=seed).fit_transform(X)
    return emb, task_ids[idx // T], idx % T


# ------------------------------------------------------------------ plotting

def plot_seed_figure(seqs, task_ids, task_names, seed, arch, out_dir):
    """2x4 figure: rows = (z carry, y carry); cols = (PCA trajectory, t-SNE, cossim, delta).
    The t-SNE column is dropped when --tsne-points=0."""
    uniq = np.unique(task_ids)
    palette = sns.color_palette('rocket', len(uniq))
    use_tsne = ARGS.tsne_points > 0
    ncols = 4 if use_tsne else 3
    fig, axs = plt.subplots(2, ncols, figsize=(FIG_WIDTH_IN * ncols, FIG_HEIGHT_IN * 2))

    for row, key in enumerate(['z', 'y']):
        seq = seqs[key]                              # [N, T, D]
        N, T, D = seq.shape
        mu, comps, evr = pca_fit(seq.reshape(-1, D))
        proj = (seq.reshape(-1, D) - mu) @ comps.T
        proj = proj.reshape(N, T, 2)
        cos, dlt = step_metrics(seq)

        ax = axs[row, 0]
        fr = lambda i: i / max(T - 1, 1)             # recursion progress in [0, 1]
        for c, t in zip(palette, uniq):
            m = proj[task_ids == t].mean(0)          # [T, 2] per-task mean trajectory
            for i in range(T - 1):
                ax.plot(m[i:i + 2, 0], m[i:i + 2, 1], '-', color=shade(c, fr(i + 1)), lw=1.0)
            for i in range(T):
                ax.plot(m[i, 0], m[i, 1], 'o', color=shade(c, fr(i)), ms=2.5)
        ax.set_xlabel(f'PC1 ({100 * evr[0]:.1f}% var)', fontsize=FONT_PT)
        ax.set_ylabel(f'PC2 ({100 * evr[1]:.1f}% var)', fontsize=FONT_PT)
        ax.set_title(f'PCA: {key} carry', fontsize=FONT_PT)

        if use_tsne:
            print(f'  t-SNE ({key} carry, up to {ARGS.tsne_points} states) ...')
            emb, e_task, e_cycle = tsne_embed(seq, task_ids, ARGS.seed)
            ax = axs[row, 1]
            for c, t in zip(palette, uniq):
                sel = e_task == t
                ax.scatter(emb[sel, 0], emb[sel, 1],
                           color=[shade(c, fr(i)) for i in e_cycle[sel]],
                           s=2, alpha=0.25, lw=0)
                # per-task mean trajectory through embedding space, cycle by cycle
                m = np.full((T, 2), np.nan)
                for i in range(T):
                    pts = emb[sel & (e_cycle == i)]
                    if len(pts):
                        m[i] = pts.mean(0)
                for i in range(T - 1):
                    ax.plot(m[i:i + 2, 0], m[i:i + 2, 1], '-', color=shade(c, fr(i + 1)), lw=1.0)
                for i in range(T):
                    ax.plot(m[i, 0], m[i, 1], 'o', color=shade(c, fr(i)), ms=2.5)
            ax.set_xlabel('t-SNE 1', fontsize=FONT_PT)
            ax.set_ylabel('t-SNE 2', fontsize=FONT_PT)
            ax.set_title(f't-SNE: {key} carry', fontsize=FONT_PT)

        for col, (vals, ylab, name) in enumerate(
                [(cos, r'$\cos(s_i, s_{i-1})$', 'Cosine Similarity'),
                 (dlt, r'$\|s_i - s_{i-1}\|$', 'State Difference')], start=ncols - 2):
            ax = axs[row, col]
            steps = np.arange(1, T)
            for c, t in zip(palette, uniq):
                v = vals[task_ids == t]
                m_t = v.mean(0)
                ax.plot(steps, m_t, color=c, lw=0.8, alpha=0.55)
                if ARGS.episodes > 1:
                    # Per-task spread over transitions; only meaningful with enough
                    # eval data per task, so gated on multi-episode runs.
                    s_t = v.std(0)
                    ax.fill_between(steps, m_t - s_t, m_t + s_t, color=c, alpha=0.10, lw=0)
            m, s = vals.mean(0), vals.std(0)
            ax.plot(steps, m, color='k', lw=2.0, zorder=5)
            ax.fill_between(steps, m - s, m + s, color='k', alpha=0.12, zorder=4)
            ax.set_xlabel('recursion step $i$', fontsize=FONT_PT)
            ax.set_ylabel(ylab, fontsize=FONT_PT)
            ax.set_xticks(steps)
            ax.set_title(f'{name}: {key} carry', fontsize=FONT_PT)

    for ax in axs.flat:
        ax.tick_params(labelsize=FONT_PT - 1)
        sns.despine(ax=ax)
    task_handles = [Line2D([], [], color=c, marker='o', ms=3, lw=1.2) for c in palette]
    task_leg = fig.legend(task_handles, [task_names[t] for t in uniq], loc='upper left',
                          bbox_to_anchor=(1.0, 0.82), fontsize=FONT_PT - 3, frameon=False,
                          title='task', title_fontsize=FONT_PT - 2)
    fig.suptitle(f'{ARCH_NAME.get(arch, arch)} Latent Analysis '
                 f'(latent_dim={seqs["ld"]}, H={seqs["H"]}, L={seqs["L"]}, seed={seed})',
                 fontsize=FONT_PT + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    # Continuous recursion-progress colorbar (light -> dark), above the task legend.
    # Added after tight_layout so the layout pass never moves it.
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'cycle_shade', [shade('#3b3b3b', 0.0), shade('#3b3b3b', 1.0)])
    # Center the colorbar horizontally on the task legend below it.
    fig.canvas.draw()
    leg_bb = task_leg.get_window_extent(fig.canvas.get_renderer()) \
                     .transformed(fig.transFigure.inverted())
    cbar_w = 0.06
    cax = fig.add_axes([leg_bb.x0 + (leg_bb.width - cbar_w) / 2, 0.91, cbar_w, 0.012])
    cbar = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap), cax=cax,
                        orientation='horizontal')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['early', 'late'])
    cbar.ax.tick_params(labelsize=FONT_PT - 3, length=0)
    cax.set_title('recursion step', fontsize=FONT_PT - 2, pad=4)
    cbar.outline.set_visible(False)
    fig.savefig(out_dir / f'recursion_pca_seed{seed}.svg', bbox_inches='tight')
    plt.close(fig)


def plot_seed_summary(per_seed, meta, out_dir):
    """2x2 cross-seed figure: rows = (z, y) carry, cols = (cossim, delta); one line per seed."""
    fig, axs = plt.subplots(2, 2, figsize=(FIG_WIDTH_IN * 2, FIG_HEIGHT_IN * 2))
    palette = sns.color_palette('rocket', len(per_seed))
    for row, key in enumerate(['z', 'y']):
        for col, (metric, name) in enumerate([('cossim', 'Cosine Similarity'),
                                              ('delta', 'State Difference')]):
            ax = axs[row, col]
            for c, (seed, data) in zip(palette, sorted(per_seed.items())):
                vals = data[key][metric]              # [N, T-1]
                steps = np.arange(1, vals.shape[1] + 1)
                m, s = vals.mean(0), vals.std(0)
                ax.plot(steps, m, color=c, lw=1.5, label=f'seed {seed}')
                ax.fill_between(steps, m - s, m + s, color=c, alpha=0.12)
            ax.set_xlabel('recursion step $i$', fontsize=FONT_PT)
            ax.set_ylabel(r'$\cos(s_i, s_{i-1})$' if metric == 'cossim' else r'$\|s_i - s_{i-1}\|$',
                          fontsize=FONT_PT)
            ax.set_xticks(np.arange(1, vals.shape[1] + 1))
            ax.set_title(f'{name}: {key} carry', fontsize=FONT_PT)
            ax.tick_params(labelsize=FONT_PT - 1)
            sns.despine(ax=ax)
    axs[0, 0].legend(fontsize=FONT_PT - 2, frameon=False)
    fig.suptitle(f'{ARCH_NAME.get(meta["arch"], meta["arch"])} Latent Analysis '
                 f'(latent_dim={meta["ld"]}, H={meta["H"]}, L={meta["L"]})',
                 fontsize=FONT_PT + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / 'recursion_metrics_seeds.svg', bbox_inches='tight')
    plt.close(fig)


# ------------------------------------------------------------------ main

def main():
    import wandb
    api = wandb.Api()
    if ARGS.group:
        runs = list(api.runs(f'{ARGS.entity}/{ARGS.project}', filters={'group': ARGS.group}))
    else:
        runs = list(api.runs(f'{ARGS.entity}/{ARGS.project}', filters={'display_name': ARGS.run}))
    assert runs, 'No matching wandb runs found.'
    group = ARGS.group or (runs[0].group or runs[0].name)
    print(f'Found {len(runs)} run(s) for {group!r}: {[r.name for r in runs]}')

    # Different eval depths get their own directory so existing plots are kept.
    suffix = f'_ep{ARGS.episodes}' if ARGS.episodes != 1 else ''
    out_dir = Path(ARGS.out_dir) if ARGS.out_dir else \
        REPO_ROOT / 'analysis' / 'recursion_pca' / (re.sub(r'[^A-Za-z0-9_.-]', '_', group) + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_rows, reward_rows = [], []
    per_seed_summary = {}
    for run in runs:
        run_config = parse_run_config(run)
        seed = run_config.get('seed', 0)
        print(f'\n=== {run.name} (seed {seed}) ===')
        artifact = latest_model_artifact(run)
        if artifact is None:
            print('  no model artifact logged yet; skipping.')
            continue
        print(f'  latest checkpoint: {artifact.name}')
        ckpt = download_checkpoint(artifact)

        cfg = build_cfg(run_config)
        set_seed(ARGS.seed)
        agent = build_agent(cfg, ckpt)
        tasks = task_vector(cfg)
        step = checkpoint_step(artifact, cfg)

        obs, act, task, rewards = get_transitions(cfg, agent, tasks, step, artifact.name)
        task_names = list(cfg.global_tasks)
        for t, r in sorted(rewards.items()):
            print(f'    {task_names[t]:<28s} episode_reward = {r:8.1f}')
            reward_rows.append((seed, task_names[t], r))

        rng = np.random.default_rng(ARGS.seed)
        z_seq, y_seq, task_ids = record_recursion(cfg, agent, obs, act, task, rng)
        print(f'  recorded recursion for {len(task_ids)} transitions: '
              f'z {z_seq.shape}, y {y_seq.shape}')

        seqs = {'z': z_seq, 'y': y_seq, 'H': cfg.H_cycles, 'L': cfg.L_cycles,
                'ld': cfg.latent_dim}
        plot_seed_figure(seqs, task_ids, task_names, seed, cfg.use_trm_dynamics, out_dir)
        meta = {'arch': cfg.use_trm_dynamics, 'ld': cfg.latent_dim,
                'H': cfg.H_cycles, 'L': cfg.L_cycles}

        per_seed_summary[seed] = {}
        for key, seq in (('z', z_seq), ('y', y_seq)):
            cos, dlt = step_metrics(seq)
            per_seed_summary[seed][key] = {'cossim': cos, 'delta': dlt}
            for t in np.unique(task_ids):
                sel = task_ids == t
                for i in range(cos.shape[1]):
                    metric_rows.append((seed, key, task_names[t], i + 1,
                                        cos[sel, i].mean(), cos[sel, i].std(),
                                        dlt[sel, i].mean(), dlt[sel, i].std()))

        del agent
        torch.cuda.empty_cache()

    assert per_seed_summary, 'No runs produced results.'
    plot_seed_summary(per_seed_summary, meta, out_dir)

    import csv
    with open(out_dir / 'recursion_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'carry', 'task', 'step', 'cossim_mean', 'cossim_sd', 'delta_mean', 'delta_sd'])
        w.writerows(metric_rows)
    with open(out_dir / 'eval_rewards.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'task', 'episode_reward'])
        w.writerows(reward_rows)
    print(f'\nDone. Outputs in {out_dir}')


if __name__ == '__main__':
    main()
