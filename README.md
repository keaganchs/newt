<h1>Tiny Recursive World Models</h1>

Code for the Master's thesis **Tiny Recursive World Models** by Keagan Holmes.

This work extends [Newt](https://www.nicklashansen.com/NewtWM), a language-conditioned
massively-multitask world model built on [TD-MPC2](https://www.tdmpc2.com), by replacing
its MLP latent-dynamics model with a **Tiny Recursive Model (TRM)**: a weight-tied
recursive network that refines the next-latent-state prediction over several
inner/outer recursion cycles, enabling cross-task knowledge sharing and
inference-time depth scaling in model-based RL. It is built on top of the official
MMBench & Newt codebase (Hansen et al., 2025).

<img src="assets/0.gif" width="12.5%"><img src="assets/1.gif" width="12.5%"><img src="assets/2.gif" width="12.5%"><img src="assets/3.gif" width="12.5%"><img src="assets/4.gif" width="12.5%"><img src="assets/5.gif" width="12.5%"><img src="assets/6.gif" width="12.5%"><img src="assets/7.gif" width="12.5%"></br>

[[Newt Website]](https://www.nicklashansen.com/NewtWM) [[Newt Paper]](https://www.nicklashansen.com/NewtWM/newt.pdf) [[Models]](https://huggingface.co/nicklashansen/newt) [[Dataset]](https://huggingface.co/datasets/nicklashansen/mmbench)

----

## Citation

If you use this work, please cite the thesis:

```
@mastersthesis{Holmes2026TRWM,
    title  = {Tiny Recursive World Models},
    author = {Keagan Holmes},
    year   = {2026},
    school = {Technical University of Darmstadt},
    type   = {Master's thesis},
}
```

This work builds directly on Newt and the MMBench task suite; please also cite:

```
@misc{Hansen2025Newt,
    title={Learning Massively Multitask World Models for Continuous Control},
    author={Nicklas Hansen and Hao Su and Xiaolong Wang},
    year={2025},
    eprint={2511.19584},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2511.19584},
}
```

----

## What this thesis adds

Newt's dynamics model is a single MLP mapping `(z_t, a_t) -> z_{t+1}` in latent space.
This thesis replaces (or augments) it with recursive alternatives that unroll a
weight-tied core for `H` outer × `L` inner cycles, and adds the training,
regularization, and analysis machinery to study them. Everything is gated behind
config flags, so the original Newt behaviour is recovered by default.

### Recursive architectures (`tdmpc2/common/trm/`)

| Module | Class(es) | Role |
|--------|-----------|------|
| `trm.py` | `TRM`, `TRMInner`, `TRMBlock`, `TRMReasoningModule` | Full Tiny Recursive Model — a looped transformer with attention + SwiGLU over tokens (task-embedding prefix tokens, patched observations), usable as an **encoder** and/or **dynamics** model. |
| `simple_trm.py` | `SimpleTRM` | Simplified two-carry dynamics head: a refined carry `y` and a working carry `z` are recursed over `(x, y, z)` triplets with a shared SimNorm MLP core — no patching/attention. The main dynamics contribution. |
| `simple_recursion_model.py` | `SRM` | Single-carry recursion (Liao & Poggio) with a learned per-cycle context/skip signal and truncated BPTT (`srm_truncation_length`). |
| `looped_transformer.py` | looped-transformer variant | Weight-tied looped transformer used in the encoder/dynamics recursion studies. |
| `trm_layers.py` | `RotaryEmbedding`, `SwiGLU`, `Attention`, `CastedLinear/Embedding` | Shared building blocks (RoPE, SwiGLU, casted params). |
| `dis_utils.py` | `advantage_margin`, `dis_beta`, … | Deep intermediate supervision (DIS) targets and the advantage-margin diagnostics. |

Selection is via config: `use_trm_dynamics ∈ {None, "simple", "trm", "srm"}` and
`use_trm_encoder`. Recursion depth is `H_cycles` × `L_cycles`, with core depth
`L_layers`; sizes come from the `TRM_SIZE` presets (S/M/L/XL) in
`tdmpc2/common/__init__.py`.

### Training & regularization additions (`tdmpc2/config.py`)

- **Latent regularization** `wm_regularization_type`: `"simnorm"` (simplicial
  embedding), `"sigreg"` (Sketched Isotropic Gaussian Regularization — an
  Epps–Pulley isotropic-Gaussian penalty replacing SimNorm; `sigreg_coef`,
  `sigreg_knots`, `sigreg_num_proj`), or `"none"`.
- **SimpleTRM skip connections** `use_simple_trm_skip_connections` /
  `simple_trm_skip_type ∈ {additive, mlp, swiglu}` (residuals across recursion cycles).
- **FiLM task conditioning** for the dynamics core (`use_film_dynamics`,
  `film_action_conditioning`).
- **Deep intermediate supervision** `use_dis_loss` / `dis_schedule` (per-outer-cycle
  supervision toward interpolated intermediate targets).
- **Inference-time depth override** `planning_H_cycles` / `planning_L_cycles` — run
  the SimpleTRM dynamics at a different depth during MPPI planning than at training
  (the dominant acting-time compute lever).
- **Diagnostics** `log_trm_gradnorms` — per-cycle gradient norms / advantage margins
  / high-carry deltas through the recursion.

### Example usage

```bash
# SimpleTRM dynamics (size L: 384-d latent, 4 outer × 3 inner cycles), SIGReg latent
python train.py use_trm_dynamics=simple trm_size=L wm_regularization_type=sigreg

# SRM dynamics with truncated BPTT
python train.py use_trm_dynamics=srm srm_truncation_length=3

# Full TRM as the encoder
python train.py use_trm_encoder=True trm_size=L

# Cheaper planning: run the recursive core at reduced depth during MPPI
python train.py use_trm_dynamics=simple planning_H_cycles=1 planning_L_cycles=1
```

### Added scripts

- **`analysis/`** — `analyze_recursion.py` (per-cycle latent-space analysis: PCA /
  t-SNE trajectories, cosine similarity, state deltas for the recursion carries;
  see `analysis/README.md`), `bench_speed.py` (inference-speed vs. recursion depth),
  `sweep_ood_cycles.py` (out-of-distribution recursion-depth sweep at inference),
  `combine_baselines.py`, `_agent_utils.py`.
- **`benchmarks/`** — `bench_concurrency.py` (how many runs fit on a GPU and what
  binds), `profile_components.py` (per-component / per-cycle wall-clock breakdown),
  plus reproduction/profiling scripts `repro_trm_*`, `investigate_trm_mem.py`,
  `bench_precision.py`, `smoke_amp.py` (see `benchmarks/README.md`).
- **`experiments/paper/`** — SLURM/shell launchers for the thesis experiments:
  `trm_{16,128,384}ld/`, `srm/`, `sigreg/`, `film/`, `maskx/`, `gradnorm/`,
  `speedtest/`, and the `42/` ablation grids (L-layers, regularization, …).

### Figures

Figures for the thesis/paper are produced from the CSVs these scripts emit by the
plotting code in a fork of the Fanda repo, under the `trwm/` directory:
**https://github.com/keaganchs/fanda/tree/trwm**

----

## MMBench

MMBench contains a total of **200** unique continuous control tasks for training of massively multitask RL policies. The task suite consists of 159 existing tasks proposed in previous work, 22 new tasks and task variants for these existing domains, as well as 19 entirely new arcade-style tasks that we dub *MiniArcade*. MMBench tasks span multiple domains and embodiments, and each task comes with language instructions, demonstrations, and optionally image observations, enabling research on both multitask pretraining, offline-to-online RL, and RL from scratch.

<img src="assets/0.png" width="100%" style="max-width: 640px"><br/>


## Newt

Newt is a language-conditioned multitask world model based on [TD-MPC2](https://www.tdmpc2.com). We train Newt by first pretraining on demonstrations to acquire task-aware representations and action priors, and then jointly optimizing with online interaction across all tasks. To extend TD-MPC2 to the massively multitask online setting, we propose a series of algorithmic improvements including a refined architecture, model-based pretraining on the available demonstrations, additional action supervision in RL policy updates, and a drastically accelerated training pipeline.

<img src="assets/1.png" width="100%" style="max-width: 640px"><br/>

----

## Getting started

We provide two options for getting started with our codebase: (1) local installation using `conda`, or (2) building a `docker` image using our provided `Dockerfile`.

First, we recommend downloading required ManiSkill assets from huggingface by running

```
wget https://huggingface.co/datasets/nicklashansen/mmbench/resolve/main/maniskill.tar.gz
tar -xvf maniskill.tar.gz && mv .maniskill ~ && rm maniskill.tar.gz
```

which will create a `.maniskill` folder in your home directory. This is the default location where the ManiSkill environments look for assets. You can also specify a different location by setting the `MANISKILL_ASSET_DIR` environment variable.

Then, choose one of the following installation options:

### Option 1: Local installation with conda

All dependencies (including the plotting/analysis stack and the Box2D
environments) are specified in `docker/environment.yaml`, so creating and
activating the environment is all that is needed:

```
conda env create -f docker/environment.yaml
conda activate newt
```

NOTE: Atari environments are currently disabled due to versioning issues. These tasks can be run by installing `gymnasium<=0.27.1` and `ale_py==0.11.2`, then disabling the maniskill tasks.


Finally, we recommend setting the `MS_SKIP_ASSET_DOWNLOAD_PROMPT` environment variable to `1` to avoid prompts from ManiSkill about downloading assets during runtime (assuming you have already downloaded the assets as described above):

```
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
```


### Option 2: Building a docker image

We provide a `Dockerfile` for easy installation. You can build the docker image by first moving your downloaded `.maniskill` asset directory to `docker/.maniskill` and then running

```
cd docker && docker build . -t <user>/newt:1.0.0
```

This docker image contains all dependencies needed for running MMBench and Newt.

----

## Example usage

### Training

Agents can be trained by running the `train.py` script. Below are some example commands:

```
$ python train.py    # <-- default: model_size=S over the DMControl task group (task=dmcontrol)
$ python train.py task=soup model_size=L    # <-- a 20M parameter agent over all 200 MMBench tasks
$ python train.py model_size=XL    # <-- an 80M parameter agent
$ python train.py model_size=S task=walker-walk   # <-- a 2M parameter single-task agent
$ python train.py obs=rgb    # <-- train with state+RGB observations
$ python train.py checkpoint=<path>/<to>/<checkpoint>.pt    # <-- resume training from checkpoint
```

Valid model sizes are `S`, `M`, `L`, and `XL` (see `common/__init__.py`). We recommend `model_size=L` (20M) for multitask experiments and `model_size=S` (2M) for single-task experiments. Note that the full `task=soup` run and several MMBench suites (ManiSkill, MetaWorld) require additional dependencies that are commented out in `docker/environment.yaml`; the default `task=dmcontrol` runs out of the box. See `config.py` for a full list of arguments (including the recursive-dynamics flags described above).

If you would like to load one of the provided Newt model checkpoints, you can download them from the [Hugging Face Models page](https://huggingface.co/nicklashansen/newt) and specify the path to the checkpoint using the `checkpoint` argument. Multitask checkpoints use a `soup` prefix in the filename, and model size is also specified in the filename. Note that this fork's model sizes are `S`/`M`/`L`/`XL` (there is no `B`), and checkpoints are not necessarily interchangeable with the original Newt release once the recursive-dynamics or SIGReg changes are enabled.

### Generating demonstrations

You can generate demonstrations using a trained agent by running the `generate_demos.py` script. You will need to specify your checkpoint directory (`CHECKPOINT_PATH`) directly in the script, as well as `data_dir` (where to save the demos), `+num_demos` (number of successful demos to collect), and `task` (task to generate demos for). Below is an example command:

```
$ python generate_demos.py task=walker-walk +num_demos=10 data_dir=<path>/<to>/<data>
```

The script assumes that the agent used for generating demos is a single-task agent trained with default hyperparameters (e.g., any of the provided checkpoints).

----

## License

This project is licensed under the MIT License - see the `LICENSE` file for details. Note that the repository relies on third-party code, which is subject to their respective licenses.
