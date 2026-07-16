from typing import Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"


def latent_act(cfg):
	"""Activation applied to encoder/dynamics latent outputs.

	With "simnorm" regularization the latent is projected onto simplices (which implicitly
	prevents representation collapse). With "sigreg" the latent is left unconstrained
	(identity) so that the SIGReg loss can regularize its distribution toward an isotropic
	Gaussian instead. With None/"none" the latent is likewise left unconstrained (identity)
	but with no regularizer at all — SimNorm and SIGReg are mutually exclusive, and only
	"simnorm" applies the activation. The surrounding NormedLinear's LayerNorm is retained
	in all cases.
	"""
	if cfg.wm_regularization_type == "simnorm":
		return SimNorm(cfg)
	return nn.Identity()


class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation.
	"""

	def __init__(self, *args, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act

	def forward(self, x):
		x = super().forward(x)
		return self.act(self.ln(x))

	def __repr__(self):
		if isinstance(self.act, nn.Sequential):
			act = '[' + ', '.join([m.__class__.__name__ for m in self.act]) + ']'
		else:
			act = self.act.__class__.__name__
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}, "\
			f"act={act})"


def mlp(in_dim, mlp_dims, out_dim, act=None):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1]))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def _core_hidden_dims(cfg):
	"""Hidden-layer widths for the recursive dynamics core (SimpleTRM / SRM).

	`L_layers` sets the DEPTH of the per-step network f -- analogous to TRM
	stacking `L_layers` reasoning blocks (see common/trm/trm.py). Passed to
	`mlp(...)`, a list of length L_layers-1 yields L_layers NormedLinear layers
	total (hidden layers use Mish; the final layer carries the latent activation).
	So L_layers=1 is a single projection, L_layers=2 a 2-layer MLP, etc.

	`xl_dynamics_mlp` keeps its historical wide [512, 512] core and takes
	precedence (it predates L_layers wiring). Hidden width otherwise follows
	latent_dim, matching the core's output width.
	"""
	if cfg.xl_dynamics_mlp:
		return [512, 512]
	return [cfg.latent_dim] * max(cfg.L_layers - 1, 0)


class FiLM(nn.Module):
	"""
	Feature-wise Linear Modulation (Perez et al., 2017): y = (1 + gamma(cond)) * x + beta(cond).
	gamma is predicted as an offset from a baseline of 1 (as in the official FiLM
	implementation), so modulation starts near identity instead of near-zeroing features.
	compute()/modulate() are split so the cond projections can be computed once per
	forward pass and reused across recursion cycles while cond is unchanged.
	"""

	def __init__(self, cond_dim: int, feature_dim: int):
		super().__init__()
		self.gamma = nn.Linear(cond_dim, feature_dim)
		self.beta = nn.Linear(cond_dim, feature_dim)

	def compute(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.gamma(cond), self.beta(cond)

	@staticmethod
	def modulate(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
		return (1.0 + gamma) * x + beta

	def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
		gamma, beta = self.compute(cond)
		return self.modulate(x, gamma, beta)

	def __repr__(self):
		return f"FiLM(cond_dim={self.gamma.in_features}, feature_dim={self.gamma.out_features})"


class FiLMBlock(nn.Module):
	"""
	Linear -> LayerNorm -> FiLM -> activation, mirroring the FiLM-ed residual blocks of
	Perez et al. (2017): modulation lands after normalization and before the
	nonlinearity, with an unmodulated residual path around the block when widths match.
	gamma/beta arrive precomputed (see FiLMedMLP.compute_film).
	"""

	def __init__(self, in_dim: int, out_dim: int, act=None, residual: bool = False):
		super().__init__()
		assert not residual or in_dim == out_dim, "residual FiLMBlock requires in_dim == out_dim"
		self.linear = nn.Linear(in_dim, out_dim)
		self.ln = nn.LayerNorm(out_dim)
		self.act = nn.Mish(inplace=False) if act is None else act
		self.residual = residual

	def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
		h = self.act(FiLM.modulate(self.ln(self.linear(x)), gamma, beta))
		return x + h if self.residual else h

	def __repr__(self):
		return f"FiLMBlock(in_features={self.linear.in_features}, "\
			f"out_features={self.linear.out_features}, "\
			f"act={self.act.__class__.__name__}, residual={self.residual})"


class FiLMedMLP(nn.Module):
	"""
	FiLM-conditioned counterpart of mlp(): every layer is a FiLMBlock with its own
	gamma/beta projections from cond, so conditioning reaches each layer pre-activation
	as in the original paper. Hidden layers use Mish and a residual path where widths
	match; the final layer carries `act` (the latent activation) and no residual, so
	constrained outputs (e.g. SimNorm) are preserved.

	Call compute_film(cond) once per forward pass (or whenever cond changes) and pass
	the result to forward() -- the cond projections are not recomputed per call.
	"""

	def __init__(self, in_dim, mlp_dims, out_dim, cond_dim, act=None):
		super().__init__()
		if isinstance(mlp_dims, int):
			mlp_dims = [mlp_dims]
		dims = [in_dim] + list(mlp_dims) + [out_dim]
		blocks = [FiLMBlock(dims[i], dims[i+1], residual=dims[i] == dims[i+1])
			for i in range(len(dims) - 2)]
		blocks.append(FiLMBlock(dims[-2], dims[-1], act=act if act else nn.Identity()))
		self.blocks = nn.ModuleList(blocks)
		self.films = nn.ModuleList([FiLM(cond_dim, d) for d in dims[1:]])

	def compute_film(self, cond: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
		return tuple(film.compute(cond) for film in self.films)

	def forward(self, x: torch.Tensor, film_params) -> torch.Tensor:
		for block, (gamma, beta) in zip(self.blocks, film_params):
			x = block(x, gamma, beta)
		return x


class FiLMDynamics(nn.Module):
	"""
	Dynamics model with FiLM task conditioning (non-recursive Newt baseline), built on
	FiLMedMLP: the conditioning signal modulates every layer pre-activation. With
	film_action_conditioning the signal is [task_emb, action] and the trunk consumes
	only z; otherwise task_emb alone modulates and the action joins the trunk input.
	xl_dynamics_mlp sizes the core [512, 512] instead of the single mlp_dim hidden layer.

	Accepts the same concatenated input as the default dynamics model:
	x = [z (latent_dim) | task_emb (task_dim) | action (action_dim)]
	"""

	def __init__(self, cfg):
		super().__init__()
		self.latent_dim = cfg.latent_dim
		self.task_dim = cfg.task_dim
		self.action_cond = cfg.film_action_conditioning
		assert self.action_cond or cfg.task_dim > 0, (
			"use_film_dynamics with use_task_embedding=False requires "
			"film_action_conditioning=True: otherwise the FiLM conditioner "
			"has no inputs and silently degrades to a learned bias."
		)
		hidden_dims = [512, 512] if cfg.xl_dynamics_mlp else [cfg.mlp_dim]
		in_dim = cfg.latent_dim + (0 if self.action_cond else cfg.action_dim)
		cond_dim = cfg.task_dim + (cfg.action_dim if self.action_cond else 0)
		self.core = FiLMedMLP(in_dim, hidden_dims, cfg.latent_dim,
			cond_dim=cond_dim, act=latent_act(cfg))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		z = x[..., :self.latent_dim]
		task_emb = x[..., self.latent_dim:self.latent_dim + self.task_dim]
		action = x[..., self.latent_dim + self.task_dim:]
		if self.action_cond:
			feats, cond = z, torch.cat([task_emb, action], dim=-1)
		else:
			feats, cond = torch.cat([z, action], dim=-1), task_emb
		return self.core(feats, self.core.compute_film(cond))


def policy(in_dim, mlp_dims, out_dim, act=None):
	"""
	Policy network for TD-MPC2.
	Vanilla MLP with ReLU activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(nn.Linear(dims[i], dims[i+1]))
		mlp.append(nn.ReLU())
	mlp.append(nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class QEnsemble(nn.Module):
	"""
	Vectorized ensemble of Q-networks. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		mlp_dims = 2*[cfg.mlp_dim]
		out_dim = max(cfg.num_bins, 1)
		self._Qs = nn.ModuleList([mlp(in_dim, mlp_dims, out_dim) for _ in range(cfg.num_q)])
		if cfg.compile:
			if cfg.rank == 0:
				print('Compiling QEnsemble forward...')
			self._forward = torch.compile(self._forward_impl, mode='reduce-overhead')
		else:
			self._forward = self._forward_impl
	
	def _forward_impl(self, x):
		outs = [q(x) for q in self._Qs]
		return torch.stack(outs, dim=0)

	def forward(self, x):
		return self._forward(x)


class QOnlineTargetEnsemble(nn.Module):
	"""
	Online and target Q-ensembles for TD-MPC2. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.online = QEnsemble(cfg)
		self.target = deepcopy(self.online)
		self.tau = cfg.tau
		self.target.train(False)
		self.track_grad(False, network='target')

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		self.online.train(mode)
		self.target.train(False)
		return self
	
	def track_grad(self, mode=True, network='online'):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		assert network in {'online', 'target'}
		module = self.online if network == 'online' else self.target
		for p in module.parameters():
			p.requires_grad_(mode)

	@torch.no_grad()
	def hard_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.copy_(op.data)

	@torch.no_grad()
	def soft_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.lerp_(op.data, self.tau)

	def forward(self, x, target=False):
		if target:
			return self.target(x)
		else:
			return self.online(x)
		

def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if cfg.use_trm_encoder:
		from common.trm import TRM
		out['state'] = TRM(cfg, model_type="encoder").to(torch.device('cuda'))
	else:
		if cfg.obs == 'state':
			out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=latent_act(cfg))
		elif cfg.obs == 'rgb':
			out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim + cfg.obs_shape['rgb'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=latent_act(cfg))
		else:
			raise NotImplementedError(f"Unexpected observation type: {cfg.obs}")
	return nn.ModuleDict(out)


def dyn(cfg, out={}):
	"""
	Returns a dynmaics model for TD-MPC2.
	"""
	if cfg.use_trm_dynamics == "trm":
		from common.trm import TRM
		out = TRM(cfg, model_type="dynamics").to(torch.device('cuda'))
	elif cfg.use_trm_dynamics == "simple":
		from common.trm.simple_trm import SimpleTRM
		out = SimpleTRM(cfg).to(torch.device('cuda'))
	elif cfg.use_trm_dynamics == "srm":
		from common.trm.simple_recursion_model import SRM
		out = SRM(cfg).to(torch.device('cuda'))
	elif cfg.use_film_dynamics:
		out = FiLMDynamics(cfg)
	else:
		hidden_dims = [512, 512] if cfg.xl_dynamics_mlp else []
		out = mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, hidden_dims, cfg.latent_dim, act=latent_act(cfg))

	return out


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Attempts to automatically convert a model checkpoint (e.g. add/remove DDP 'module.' prefixes).
	"""
	encoder_key = 'module._encoder.state.0.weight'
	if encoder_key in source_state_dict and encoder_key not in target_state_dict:
		# Remove 'module.' prefix from all keys in source_state_dict
		source_state_dict = {k[len('module.'):]: v for k, v in source_state_dict.items()}
	if encoder_key in target_state_dict and encoder_key not in source_state_dict:
		# Add 'module.' prefix to all keys in source_state_dict
		source_state_dict = {'module.' + k: v for k, v in source_state_dict.items()}

	for key in ['_encoder.state.0.weight', 'module._encoder.state.0.weight']:
		if key in target_state_dict and key in source_state_dict and \
				target_state_dict[key].shape != source_state_dict[key].shape:
			# possible rgb input in target but not in source, we should pad
			print('Warning: unexpected shape mismatch in encoder weights, attempting to pad source weights...')
			pad = target_state_dict[key].shape[1] - source_state_dict[key].shape[1]
			assert pad > 0, f'pad f({pad}) should be positive'
			pad_tensor = torch.zeros(source_state_dict[key].shape[0], pad, device=source_state_dict[key].device)
			source_state_dict[key] = torch.cat([source_state_dict[key], pad_tensor], dim=1)

	if '_action_masks' in target_state_dict and '_action_masks' in source_state_dict and \
			source_state_dict['_action_masks'].shape != target_state_dict['_action_masks'].shape:
		# repeat first dimension to match
		source_state_dict['_action_masks'] = source_state_dict['_action_masks'].repeat(
			target_state_dict['_action_masks'].shape[0] // source_state_dict['_action_masks'].shape[0], 1)
		if '_task_emb.weight' in source_state_dict:
			source_state_dict['_task_emb.weight'] = source_state_dict['_task_emb.weight'].repeat(
				target_state_dict['_action_masks'].shape[0] // source_state_dict['_task_emb.weight'].shape[0], 1)
		
	if '_task_emb.weight' in source_state_dict and not '_task_emb.weight' in target_state_dict:
		# delete task embedding from source state dict
		source_state_dict.pop('_task_emb.weight', None)

	return source_state_dict


def print_mismatched_tensors(target_state_dict, source_state_dict):
	target_keys = set(target_state_dict.keys())
	source_keys = set(source_state_dict.keys())

	# Keys in source but not in target
	for key in source_keys - target_keys:
		print(f"[Extra in source] {key}: shape={tuple(source_state_dict[key].shape)}")

	# Keys in target but not in source
	for key in target_keys - source_keys:
		print(f"[Missing in source] {key}: expected shape={tuple(target_state_dict[key].shape)}")

	# Keys present in both but with shape mismatch
	for key in target_keys & source_keys:
		try:
			t_shape = tuple(target_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in target_state_dict] {key}: {e}")
			continue
		try:
			s_shape = tuple(source_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in source_state_dict] {key}: {e}")
			continue
		if t_shape != s_shape:
			print(f"[Shape mismatch] {key}: target={t_shape}, source={s_shape}")


